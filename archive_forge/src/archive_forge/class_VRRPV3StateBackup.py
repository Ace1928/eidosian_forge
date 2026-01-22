import abc
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
class VRRPV3StateBackup(VRRPState):

    def _master_down(self):
        vrrp_router = self.vrrp_router
        vrrp_router.send_advertisement()
        vrrp_router.preempt_delay_timer.cancel()
        vrrp_router.state_change(vrrp_event.VRRP_STATE_MASTER)
        vrrp_router.adver_timer.start(vrrp_router.config.advertisement_interval)

    def master_down(self, ev):
        self.vrrp_router.logger.debug('%s master_down', self.__class__.__name__)
        self._master_down()

    def adver(self, ev):
        vrrp_router = self.vrrp_router
        vrrp_router.logger.debug('adver %s %s %s', self.__class__.__name__, ev.__class__.__name__, vrrp_router.state)

    def preempt_delay(self, ev):
        self.vrrp_router.logger.warning('%s preempt_delay', self.__class__.__name__)
        self._master_down()

    def vrrp_received(self, ev):
        vrrp_router = self.vrrp_router
        vrrp_router.logger.debug('%s vrrp_received', self.__class__.__name__)
        _ip, vrrp_ = vrrp.vrrp.get_payload(ev.packet)
        if vrrp_.priority == 0:
            vrrp_router.master_down_timer.start(vrrp_router.params.skew_time)
        else:
            params = vrrp_router.params
            config = vrrp_router.config
            if not config.preempt_mode or config.priority <= vrrp_.priority:
                params.master_adver_interval = vrrp_.max_adver_int_in_sec
                vrrp_router.master_down_timer.start(params.master_down_interval)
            elif config.preempt_mode and config.preempt_delay > 0 and (config.priority > vrrp_.priority):
                if not vrrp_router.preempt_delay_timer.is_running():
                    vrrp_router.preempt_delay_timer.start(config.preempt_delay)
                vrrp_router.master_down_timer.start(params.master_down_interval)

    def vrrp_shutdown_request(self, ev):
        vrrp_router = self.vrrp_router
        vrrp_router.logger.debug('%s vrrp_shutdown_request', self.__class__.__name__)
        vrrp_router.preempt_delay_timer.cancel()
        vrrp_router.master_down_timer.cancel()
        vrrp_router.state_change(vrrp_event.VRRP_STATE_INITIALIZE)

    def vrrp_config_change_request(self, ev):
        vrrp_router = self.vrrp_router
        vrrp_router.logger.warning('%s vrrp_config_change_request', self.__class__.__name__)
        if ev.priority is not None and vrrp_router.config.address_owner:
            vrrp_router.master_down_timer.cancel()
            self._master_down()
        if ev.preempt_mode is not None or ev.preempt_delay is not None:
            vrrp_router.preempt_delay_timer.cancel()