import abc
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
def adver(self, ev):
    vrrp_router = self.vrrp_router
    vrrp_router.logger.debug('adver %s %s %s', self.__class__.__name__, ev.__class__.__name__, vrrp_router.state)