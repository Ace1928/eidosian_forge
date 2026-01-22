import contextlib
import logging
import os
import socket
import struct
from os_ken import cfg
from os_ken.base import app_manager
from os_ken.base.app_manager import OSKenApp
from os_ken.controller.handler import set_ev_cls
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.packet import zebra
from os_ken.services.protocols.zebra import db
from os_ken.services.protocols.zebra import event
from os_ken.services.protocols.zebra.server import event as zserver_event
class ZClient(object):
    """
    Zebra client class.
    """

    def __init__(self, server, sock, addr):
        self.server = server
        self.sock = sock
        self.addr = addr
        self.logger = server.logger
        self.is_active = False
        self._threads = []
        self.send_q = hub.Queue(16)
        self.zserv_ver = CONF.server_version
        self.route_type = None

    def start(self):
        self.is_active = True
        self.sock.settimeout(GLOBAL_CONF.socket_timeout)
        self._threads.append(hub.spawn(self._send_loop))
        self._threads.append(hub.spawn(self._recv_loop))
        self.server.send_event_to_observers(zserver_event.EventZClientConnected(self))
        hub.joinall(self._threads)
        self.server.send_event_to_observers(zserver_event.EventZClientDisconnected(self))

    def stop(self):
        self.is_active = False

    def _send_loop(self):
        try:
            while self.is_active:
                buf = self.send_q.get()
                self.sock.sendall(buf)
        except socket.error as e:
            self.logger.exception('Error while sending message to Zebra client%s: %s', self.addr, e)
        self.stop()

    def _recv_loop(self):
        buf = b''
        min_len = recv_len = zebra.ZebraMessage.get_header_size(self.zserv_ver)
        try:
            while self.is_active:
                try:
                    recv_buf = self.sock.recv(recv_len)
                except socket.timeout:
                    continue
                if len(recv_buf) == 0:
                    break
                buf += recv_buf
                while len(buf) >= min_len:
                    length, = struct.unpack_from('!H', buf)
                    if length - len(buf) > 0:
                        recv_len = length - len(buf)
                        break
                    msg, _, buf = zebra.ZebraMessage.parser(buf)
                    ev = event.message_to_event(self, msg)
                    if ev:
                        self.logger.debug('Notify event: %s', ev)
                        self.server.send_event_to_observers(ev)
        except socket.error as e:
            self.logger.exception('Error while sending message to Zebra client%s: %s', self.addr, e)
        self.stop()

    def send_msg(self, msg):
        """
        Sends Zebra message.

        :param msg: Instance of py:class: `os_ken.lib.packet.zebra.ZebraMessage`.
        :return: Serialized msg if succeeded, otherwise None.
        """
        if not self.is_active:
            self.logger.debug('Cannot send message: Already deactivated: msg=%s', msg)
            return
        elif not self.send_q:
            self.logger.debug('Cannot send message: Send queue does not exist: msg=%s', msg)
            return
        elif self.zserv_ver != msg.version:
            self.logger.debug('Zebra protocol version mismatch:server_version=%d, msg.version=%d', self.zserv_ver, msg.version)
            msg.version = self.zserv_ver
        self.send_q.put(msg.serialize())