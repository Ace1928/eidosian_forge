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
class _NetworkController(FlexinetPeer, Activity):
    """Network controller peer.

    Provides MessagePackRPC interface for flexinet peers like Network
    controller to peer and have RPC session with BGPS process. This RPC
    interface provides access to BGPS API.
    """

    def __init__(self):
        FlexinetPeer.__init__(self)
        Activity.__init__(self, name='NETWORK_CONTROLLER')
        self._outstanding_reqs = {}
        self._rpc_sessions = {}

    def _run(self, *args, **kwargs):
        """Runs RPC server.

        Wait for peer to connect and start rpc session with it.
        For every connection we start and new rpc session.
        """
        apgw_rpc_bind_ip = _validate_rpc_ip(kwargs.pop(NC_RPC_BIND_IP))
        apgw_rpc_bind_port = _validate_rpc_port(kwargs.pop(NC_RPC_BIND_PORT))
        sock_addr = (apgw_rpc_bind_ip, apgw_rpc_bind_port)
        LOG.debug('NetworkController started listening for connections...')
        server_thread, _ = self._listen_tcp(sock_addr, self._start_rpc_session)
        self.pause(0)
        server_thread.wait()

    def _start_rpc_session(self, sock):
        """Starts a new RPC session with given connection.
        """
        session_name = RpcSession.NAME_FMT % str(sock.getpeername())
        self._stop_child_activities(session_name)
        rpc_session = RpcSession(sock, self)
        self._spawn_activity(rpc_session)

    def _send_rpc_notification_to_session(self, session, method, params):
        if not session.is_connected:
            self._stop_child_activities(session.name)
            return
        return session.send_notification(method, params)

    def send_rpc_notification(self, method, params):
        if not self.started:
            return
        for session in list(self._child_activity_map.values()):
            if not isinstance(session, RpcSession):
                continue
            self._send_rpc_notification_to_session(session, method, params)