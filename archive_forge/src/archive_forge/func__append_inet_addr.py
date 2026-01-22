import logging
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from os_ken.lib import netdevice
from os_ken.lib import ip
from os_ken.lib.packet import zebra
from . import base
def _append_inet_addr(intf_inet, addr):
    addr_list = intf_inet.split(',')
    if addr in addr_list:
        LOG.debug('Interface "%s" has already "ifaddr": %s', intf.ifname, addr)
        return intf_inet
    else:
        addr_list.append(addr)
        return ','.join(addr_list)