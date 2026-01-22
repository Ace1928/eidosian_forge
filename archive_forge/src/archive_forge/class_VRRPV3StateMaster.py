import abc
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
class VRRPV3StateMaster(VRRPState):

    def master_down(self, ev):
        vrrp_router = self.vrrp_router
        vrrp_router.logger.debug('%s master_down %s %s', self.__class__.__name__, ev.__class__.__name__, vrrp_router.state)

    def _adver(self):
        vrrp_router = self.vrrp_router
        vrrp_router.send_advertisement()
        vrrp_router.adver_timer.start(vrrp_router.config.advertisement_interval)

    def adver(self, ev):
        self.vrrp_router.logger.debug('%s adver', self.__class__.__name__)
        self._adver()

    def preempt_delay(self, ev):
        self.vrrp_router.logger.warning('%s preempt_delay', self.__class__.__name__)

    def vrrp_received(self, ev):
        vrrp_router = self.vrrp_router
        vrrp_router.logger.debug('%s vrrp_received', self.__class__.__name__)
        ip, vrrp_ = vrrp.vrrp.get_payload(ev.packet)
        config = vrrp_router.config
        if vrrp_.priority == 0:
            vrrp_router.send_advertisement()
            vrrp_router.adver_timer.start(config.advertisement_interval)
        else:
            params = vrrp_router.params
            if config.priority < vrrp_.priority or (config.priority == vrrp_.priority and vrrp.ip_address_lt(vrrp_router.interface.primary_ip_address, ip.src)):
                vrrp_router.adver_timer.cancel()
                params.master_adver_interval = vrrp_.max_adver_int_in_sec
                vrrp_router.state_change(vrrp_event.VRRP_STATE_BACKUP)
                vrrp_router.master_down_timer.start(params.master_down_interval)

    def vrrp_shutdown_request(self, ev):
        vrrp_router = self.vrrp_router
        vrrp_router.logger.debug('%s vrrp_shutdown_request', self.__class__.__name__)
        vrrp_router.adver_timer.cancel()
        vrrp_router.send_advertisement(True)
        vrrp_router.state_change(vrrp_event.VRRP_STATE_INITIALIZE)

    def vrrp_config_change_request(self, ev):
        vrrp_router = self.vrrp_router
        vrrp_router.logger.warning('%s vrrp_config_change_request', self.__class__.__name__)
        if ev.priority is not None or ev.advertisement_interval is not None:
            vrrp_router.adver_timer.cancel()
            self._adver()