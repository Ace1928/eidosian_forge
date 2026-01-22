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
@set_ev_cls(event.EventZebraRouterIDAdd)
def _router_id_add_handler(self, ev):
    self.logger.debug('Client %s requests router_id, server will response: router_id=%s', ev.zclient, self.router_id)
    msg = zebra.ZebraMessage(body=zebra.ZebraRouterIDUpdate(family=socket.AF_INET, prefix='%s/32' % self.router_id))
    ev.zclient.send_msg(msg)