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
class ZServer(OSKenApp):
    """
    The base class for Zebra server application.
    """
    _EVENTS = event.ZEBRA_EVENTS + [zserver_event.EventZClientConnected, zserver_event.EventZClientDisconnected]

    def __init__(self, *args, **kwargs):
        super(ZServer, self).__init__(*args, **kwargs)
        self.zserv = None
        self.zserv_addr = (CONF.server_host, CONF.server_port)
        self.zapi_connection_family = detect_address_family(CONF.server_host)
        self.router_id = CONF.router_id

    def start(self):
        super(ZServer, self).start()
        if self.zapi_connection_family == socket.AF_UNIX:
            unix_sock_dir = os.path.dirname(CONF.server_host)
            if os.path.exists(CONF.server_host):
                os.remove(CONF.server_host)
            if not os.path.isdir(unix_sock_dir):
                os.mkdir(unix_sock_dir)
                os.chmod(unix_sock_dir, 511)
        try:
            self.zserv = hub.StreamServer(self.zserv_addr, zclient_connection_factory)
        except OSError as e:
            self.logger.error('Cannot start Zebra server%s: %s', self.zserv_addr, e)
            raise e
        if self.zapi_connection_family == socket.AF_UNIX:
            os.chmod(CONF.server_host, 511)
        self._add_lo_interface()
        return hub.spawn(self.zserv.serve_forever)

    def _add_lo_interface(self):
        intf = db.interface.ip_link_add(SESSION, 'lo')
        if intf:
            self.logger.debug('Added interface "%s": %s', intf.ifname, intf)
        route = db.route.ip_route_add(SESSION, destination='127.0.0.0/8', device='lo', source='127.0.0.1/8', route_type=zebra.ZEBRA_ROUTE_CONNECT)
        if route:
            self.logger.debug('Added route to "%s": %s', route.destination, route)

    @set_ev_cls(event.EventZebraHello)
    def _hello_handler(self, ev):
        if ev.body is None:
            self.logger.debug('Client %s says hello.', ev.zclient)
            return
        ev.zclient.route_type = ev.body.route_type
        self.logger.debug('Client %s says hello and bids fair to announce only %s routes', ev.zclient, ev.body.route_type)

    @set_ev_cls(event.EventZebraRouterIDAdd)
    def _router_id_add_handler(self, ev):
        self.logger.debug('Client %s requests router_id, server will response: router_id=%s', ev.zclient, self.router_id)
        msg = zebra.ZebraMessage(body=zebra.ZebraRouterIDUpdate(family=socket.AF_INET, prefix='%s/32' % self.router_id))
        ev.zclient.send_msg(msg)

    @set_ev_cls(event.EventZebraInterfaceAdd)
    def _interface_add_handler(self, ev):
        self.logger.debug('Client %s requested all interfaces', ev.zclient)
        interfaces = db.interface.ip_address_show_all(SESSION)
        self.logger.debug('Server will response interfaces: %s', interfaces)
        for intf in interfaces:
            msg = zebra.ZebraMessage(body=zebra.ZebraInterfaceAdd(ifname=intf.ifname, ifindex=intf.ifindex, status=intf.status, if_flags=intf.flags, ptm_enable=zebra.ZEBRA_IF_PTM_ENABLE_OFF, ptm_status=zebra.ZEBRA_PTM_STATUS_UNKNOWN, metric=intf.metric, ifmtu=intf.ifmtu, ifmtu6=intf.ifmtu6, bandwidth=intf.bandwidth, ll_type=intf.ll_type, hw_addr=intf.hw_addr))
            ev.zclient.send_msg(msg)
            routes = db.route.ip_route_show_all(SESSION, ifindex=intf.ifindex, is_selected=True)
            self.logger.debug('Server will response routes: %s', routes)
            for route in routes:
                dest, _ = route.destination.split('/')
                msg = zebra.ZebraMessage(body=zebra.ZebraInterfaceAddressAdd(ifindex=intf.ifindex, ifc_flags=0, family=None, prefix=route.source, dest=dest))
                ev.zclient.send_msg(msg)

    @set_ev_cls([event.EventZebraIPv4RouteAdd, event.EventZebraIPv6RouteAdd])
    def _ip_route_add_handler(self, ev):
        self.logger.debug('Client %s advertised IP route: %s', ev.zclient, ev.body)
        for nexthop in ev.body.nexthops:
            route = db.route.ip_route_add(SESSION, destination=ev.body.prefix, gateway=nexthop.addr, ifindex=nexthop.ifindex or 0, route_type=ev.body.route_type)
            if route:
                self.logger.debug('Added route to "%s": %s', route.destination, route)

    @set_ev_cls([event.EventZebraIPv4RouteDelete, event.EventZebraIPv6RouteDelete])
    def _ip_route_delete_handler(self, ev):
        self.logger.debug('Client %s withdrew IP route: %s', ev.zclient, ev.body)
        for nexthop in ev.body.nexthops:
            routes = db.route.ip_route_delete(SESSION, destination=ev.body.prefix, gateway=nexthop.addr, route_type=ev.body.route_type)
            if routes:
                self.logger.debug('Deleted routes to "%s": %s', ev.body.prefix, routes)