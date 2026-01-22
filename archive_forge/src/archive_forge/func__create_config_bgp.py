import logging
import os
import netaddr
from . import docker_base as base
def _create_config_bgp(self):
    c = base.CmdBuffer()
    c << 'hostname bgpd'
    c << 'password zebra'
    c << 'router bgp {0}'.format(self.asn)
    c << 'bgp router-id {0}'.format(self.router_id)
    if any((info['graceful_restart'] for info in self.peers.values())):
        c << 'bgp graceful-restart'
    version = 4
    for peer, info in self.peers.items():
        version = netaddr.IPNetwork(info['neigh_addr']).version
        n_addr = info['neigh_addr'].split('/')[0]
        if version == 6:
            c << 'no bgp default ipv4-unicast'
        c << 'neighbor {0} remote-as {1}'.format(n_addr, peer.asn)
        if info['is_rs_client']:
            c << 'neighbor {0} route-server-client'.format(n_addr)
        for typ, p in info['policies'].items():
            c << 'neighbor {0} route-map {1} {2}'.format(n_addr, p['name'], typ)
        if info['passwd']:
            c << 'neighbor {0} password {1}'.format(n_addr, info['passwd'])
        if info['passive']:
            c << 'neighbor {0} passive'.format(n_addr)
        if version == 6:
            c << 'address-family ipv6 unicast'
            c << 'neighbor {0} activate'.format(n_addr)
            c << 'exit-address-family'
    for route in self.routes.values():
        if route['rf'] == 'ipv4':
            c << 'network {0}'.format(route['prefix'])
        elif route['rf'] == 'ipv6':
            c << 'address-family ipv6 unicast'
            c << 'network {0}'.format(route['prefix'])
            c << 'exit-address-family'
        else:
            raise Exception('unsupported route faily: {0}'.format(route['rf']))
    if self.zebra:
        if version == 6:
            c << 'address-family ipv6 unicast'
            c << 'redistribute connected'
            c << 'exit-address-family'
        else:
            c << 'redistribute connected'
    for name, policy in self.policies.items():
        c << 'access-list {0} {1} {2}'.format(name, policy['type'], policy['match'])
        c << 'route-map {0} permit 10'.format(name)
        c << 'match ip address {0}'.format(name)
        c << 'set metric {0}'.format(policy['med'])
    c << 'debug bgp as4'
    c << 'debug bgp fsm'
    c << 'debug bgp updates'
    c << 'debug bgp events'
    c << 'log file {0}/bgpd.log'.format(self.SHARED_VOLUME)
    with open('{0}/bgpd.conf'.format(self.config_dir), 'w') as f:
        LOG.info("[%s's new config]", self.name)
        LOG.info(str(c))
        f.writelines(str(c))