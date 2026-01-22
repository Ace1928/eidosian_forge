import os
import logging
from os_ken.lib import hub, alert
from os_ken.base import app_manager
from os_ken.controller import event
def _accept_loop_nw_sock(self):
    self.logger.info('Network socket server start listening...')
    while True:
        conn, addr = self.nwsock.accept()
        self.logger.info('Connected with %s', addr[0])
        hub.spawn(self._recv_loop_nw_sock, conn, addr)