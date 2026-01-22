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
@set_ev_cls([event.EventZebraIPv4RouteDelete, event.EventZebraIPv6RouteDelete])
def _ip_route_delete_handler(self, ev):
    self.logger.debug('Client %s withdrew IP route: %s', ev.zclient, ev.body)
    for nexthop in ev.body.nexthops:
        routes = db.route.ip_route_delete(SESSION, destination=ev.body.prefix, gateway=nexthop.addr, route_type=ev.body.route_type)
        if routes:
            self.logger.debug('Deleted routes to "%s": %s', ev.body.prefix, routes)