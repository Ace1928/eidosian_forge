import time
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import monitor as vrrp_monitor
from os_ken.services.protocols.vrrp import router as vrrp_router
@staticmethod
def _instance_name(interface, vrid, is_ipv6):
    ip_version = 'ipv6' if is_ipv6 else 'ipv4'
    return 'VRRP-Router-%s-%d-%s' % (str(interface), vrid, ip_version)