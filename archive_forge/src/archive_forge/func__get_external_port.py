import functools
import logging
from os_ken import cfg
import os_ken.exception as os_ken_exc
import os_ken.lib.dpid as dpid_lib
import os_ken.lib.ovs.vsctl as ovs_vsctl
from os_ken.lib.ovs.vsctl import valid_ovsdb_addr
def _get_external_port(self, name):
    external_ids = self.db_get_map('Interface', name, 'external_ids')
    if external_ids:
        return
    options = self.db_get_map('Interface', name, 'options')
    if 'remote_ip' in options:
        return
    ofport = self.get_ofport(name)
    return VifPort(name, ofport, None, None, self)