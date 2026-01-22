import logging
import socket
import traceback
import msgpack
from os_ken.lib.packet import safi as subaddr_family
from os_ken.services.protocols.bgp import api
from os_ken.services.protocols.bgp.api.base import ApiException
from os_ken.services.protocols.bgp.api.base import NEXT_HOP
from os_ken.services.protocols.bgp.api.base import ORIGIN_RD
from os_ken.services.protocols.bgp.api.base import PREFIX
from os_ken.services.protocols.bgp.api.base import ROUTE_DISTINGUISHER
from os_ken.services.protocols.bgp.api.base import VPN_LABEL
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import FlexinetPeer
from os_ken.services.protocols.bgp.base import NET_CTRL_ERROR_CODE
from os_ken.services.protocols.bgp.constants import VRF_TABLE
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfConf
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv4
class RpcSession(Activity):
    """Provides message-pack RPC abstraction for one session.

    It contains message-pack packer, un-packer, message ID sequence
    and utilities that use these. It also cares about socket communication w/
    RPC peer.
    """
    NAME_FMT = 'RpcSession%s'

    def __init__(self, sock, outgoing_msg_sink_iter):
        self.peer_name = str(sock.getpeername())
        super(RpcSession, self).__init__(self.NAME_FMT % self.peer_name)
        self._packer = msgpack.Packer()
        self._unpacker = msgpack.Unpacker(strict_map_key=False)
        self._next_msgid = 0
        self._socket = sock
        self._outgoing_msg_sink_iter = outgoing_msg_sink_iter
        self.is_connected = True
        self.green_in = None
        self.green_out = None

    def stop(self):
        super(RpcSession, self).stop()
        self.is_connected = False
        LOG.info('RPC Session to %s stopped', self.peer_name)

    def _run(self):
        self.green_out = self._spawn('net_ctrl._process_outgoing', self._process_outgoing_msg, self._outgoing_msg_sink_iter)
        self.green_in = self._spawn('net_ctrl._process_incoming', self._process_incoming_msgs)
        LOG.info('RPC Session to %s started', self.peer_name)
        self.green_in.wait()
        self.green_out.wait()

    def _next_msg_id(self):
        this_id = self._next_msgid
        self._next_msgid += 1
        return this_id

    def create_request(self, method, params):
        msgid = self._next_msg_id()
        return self._packer.pack([RPC_MSG_REQUEST, msgid, method, params])

    def create_error_response(self, msgid, error):
        if error is None:
            raise NetworkControllerError(desc='Creating error without body!')
        return self._packer.pack([RPC_MSG_RESPONSE, msgid, error, None])

    def create_success_response(self, msgid, result):
        if result is None:
            raise NetworkControllerError(desc='Creating response without body!')
        return self._packer.pack([RPC_MSG_RESPONSE, msgid, None, result])

    def create_notification(self, method, params):
        return self._packer.pack([RPC_MSG_NOTIFY, method, params])

    def feed_and_get_messages(self, data):
        self._unpacker.feed(data)
        messages = []
        for msg in self._unpacker:
            messages.append(msg)
        return messages

    def feed_and_get_first_message(self, data):
        self._unpacker.feed(data)
        for msg in self._unpacker:
            return msg

    def _send_error_response(self, request, err_msg):
        rpc_msg = self.create_error_response(request[RPC_IDX_MSG_ID], str(err_msg))
        return self._sendall(rpc_msg)

    def _send_success_response(self, request, result):
        rpc_msg = self.create_success_response(request[RPC_IDX_MSG_ID], result)
        return self._sendall(rpc_msg)

    def send_notification(self, method, params):
        rpc_msg = self.create_notification(method, params)
        return self._sendall(rpc_msg)

    def _process_incoming_msgs(self):
        LOG.debug('NetworkController started processing incoming messages')
        assert self._socket
        while self.is_connected:
            msg_buff = self._recv()
            if len(msg_buff) == 0:
                LOG.info('Peer %s disconnected.', self.peer_name)
                self.is_connected = False
                self._socket.close()
                break
            messages = self.feed_and_get_messages(msg_buff)
            for msg in messages:
                if msg[0] == RPC_MSG_REQUEST:
                    try:
                        result = _handle_request(msg)
                        self._send_success_response(msg, result)
                    except BGPSException as e:
                        self._send_error_response(msg, e.message)
                elif msg[0] == RPC_MSG_RESPONSE:
                    _handle_response(msg)
                elif msg[0] == RPC_MSG_NOTIFY:
                    _handle_notification(msg)
                else:
                    LOG.error('Invalid message type: %r', msg)
                self.pause(0)
        if self.green_out:
            self.green_out.kill()

    def _process_outgoing_msg(self, sink_iter):
        """For every message we construct a corresponding RPC message to be
        sent over the given socket inside given RPC session.

        This function should be launched in a new green thread as
        it loops forever.
        """
        LOG.debug('NetworkController processing outgoing request list.')
        from os_ken.services.protocols.bgp.model import FlexinetOutgoingRoute
        while self.is_connected:
            for outgoing_msg in sink_iter:
                if not self.is_connected:
                    self._socket.close()
                    return
                if isinstance(outgoing_msg, FlexinetOutgoingRoute):
                    rpc_msg = _create_prefix_notification(outgoing_msg, self)
                else:
                    raise NotImplementedError('Do not handle out going message of type %s' % outgoing_msg.__class__)
                if rpc_msg:
                    self._sendall(rpc_msg)
            self.pause(0)
        if self.green_in:
            self.green_in.kill()

    def _recv(self):
        return self._sock_wrap(self._socket.recv)(RPC_SOCK_BUFF_SIZE)

    def _sendall(self, msg):
        return self._sock_wrap(self._socket.sendall)(msg)

    def _sock_wrap(self, func):

        def wrapper(*args, **kwargs):
            try:
                ret = func(*args, **kwargs)
            except socket.error:
                LOG.error(traceback.format_exc())
                self._socket_error()
                return
            return ret
        return wrapper

    def _socket_error(self):
        if self.started:
            self.stop()