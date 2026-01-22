import time
import random
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.lib import mac as lib_mac
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import api as vrrp_api
from os_ken.services.protocols.vrrp import event as vrrp_event
def _main_version_priority_sleep(self, vrrp_version, priority, do_sleep):
    app_mgr = app_manager.AppManager.get_instance()
    self.logger.debug('%s', app_mgr.applications)
    vrrp_mgr = app_mgr.applications['VRRPManager']
    step = 5
    instances = {}
    for vrid in range(1, 256, step):
        if vrid == _VRID:
            continue
        print('vrid %s' % vrid)
        l = {}
        prio = max(vrrp.VRRP_PRIORITY_BACKUP_MIN, min(vrrp.VRRP_PRIORITY_BACKUP_MAX, vrid))
        rep0 = self._configure_vrrp_router(vrrp_version, prio, _PRIMARY_IP_ADDRESS0, self._IFNAME0, vrid)
        assert rep0.instance_name is not None
        l[0] = rep0
        prio = max(vrrp.VRRP_PRIORITY_BACKUP_MIN, min(vrrp.VRRP_PRIORITY_BACKUP_MAX, 256 - vrid))
        rep1 = self._configure_vrrp_router(vrrp_version, prio, _PRIMARY_IP_ADDRESS1, self._IFNAME1, vrid)
        assert rep1.instance_name is not None
        l[1] = rep1
        instances[vrid] = l
    print('vrid %s' % _VRID)
    l = {}
    rep0 = self._configure_vrrp_router(vrrp_version, priority, _PRIMARY_IP_ADDRESS0, self._IFNAME0, _VRID)
    assert rep0.instance_name is not None
    l[0] = rep0
    rep1 = self._configure_vrrp_router(vrrp_version, vrrp.VRRP_PRIORITY_BACKUP_DEFAULT, _PRIMARY_IP_ADDRESS1, self._IFNAME1, _VRID)
    assert rep1.instance_name is not None
    l[1] = rep1
    instances[_VRID] = l
    self.logger.debug('%s', vrrp_mgr._instances)
    if do_sleep:
        print('priority %s' % priority)
        print('waiting for instances starting')
        self._check(vrrp_api, instances)
    for vrid in instances.keys():
        if vrid == _VRID:
            continue
        which = vrid & 1
        new_priority = int(random.uniform(vrrp.VRRP_PRIORITY_BACKUP_MIN, vrrp.VRRP_PRIORITY_BACKUP_MAX))
        i = instances[vrid][which]
        vrrp_api.vrrp_config_change(self, i.instance_name, priority=new_priority)
        i.config.priority = new_priority
    if do_sleep:
        print('priority shuffled')
        self._check(vrrp_api, instances)
    for vrid in instances.keys():
        if vrid == _VRID:
            continue
        which = vrid & 1
        vrrp_api.vrrp_shutdown(self, instances[vrid][which].instance_name)
    vrrp_api.vrrp_shutdown(self, instances[_VRID][0].instance_name)
    if do_sleep:
        print('shutting down instances')
        while True:
            rep = vrrp_api.vrrp_list(self)
            if len(rep.instance_list) <= len(instances):
                break
            print('left %s' % len(rep.instance_list))
            time.sleep(1)
        assert len(rep.instance_list) == len(instances)
        print('waiting for the rest becoming master')
        while True:
            rep = vrrp_api.vrrp_list(self)
            if all((i.state == vrrp_event.VRRP_STATE_MASTER for i in rep.instance_list)):
                break
            time.sleep(1)
    vrrp_api.vrrp_shutdown(self, instances[_VRID][1].instance_name)
    for vrid in instances.keys():
        if vrid == _VRID:
            continue
        which = 1 - (vrid & 1)
        vrrp_api.vrrp_shutdown(self, instances[vrid][which].instance_name)
    print('waiting for instances shutting down')
    while True:
        rep = vrrp_api.vrrp_list(self)
        if not rep.instance_list:
            break
        print('left %s' % len(rep.instance_list))
        time.sleep(1)