import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
class TunnelPort(object):

    def __init__(self, port_name, ofport, tunnel_type, local_ip, remote_ip):
        super(TunnelPort, self).__init__()
        self.port_name = port_name
        self.ofport = ofport
        self.tunnel_type = tunnel_type
        self.local_ip = local_ip
        self.remote_ip = remote_ip

    def __eq__(self, other):
        return self.port_name == other.port_name and self.ofport == other.ofport and (self.tunnel_type == other.tunnel_type) and (self.local_ip == other.local_ip) and (self.remote_ip == other.remote_ip)

    def __str__(self):
        return 'port_name=%s, ofport=%s, type=%s, local_ip=%s, remote_ip=%s' % (self.port_name, self.ofport, self.tunnel_type, self.local_ip, self.remote_ip)