import os
import logging
from os_ken.lib import hub, alert
from os_ken.base import app_manager
from os_ken.controller import event
def _recv_loop_nw_sock(self, conn, addr):
    buf = bytes()
    while True:
        ret = conn.recv(BUFSIZE)
        if len(ret) == 0:
            self.logger.info('Disconnected from %s', addr[0])
            break
        buf += ret
        while len(buf) >= BUFSIZE:
            data = buf[:BUFSIZE]
            msg = alert.AlertPkt.parser(data)
            if msg:
                self.send_event_to_observers(EventAlert(msg))
            buf = buf[BUFSIZE:]