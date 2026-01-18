from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def update_ddp_settings(self, check_mode=False):
    """Update dynamic disk pool settings."""
    if self.raid_level != 'raidDiskPool':
        return False
    needs_update = False
    if self.pool_detail['volumeGroupData']['diskPoolData']['poolUtilizationWarningThreshold'] != self.ddp_warning_threshold_pct or self.pool_detail['volumeGroupData']['diskPoolData']['poolUtilizationCriticalThreshold'] != self.ddp_critical_threshold_pct:
        needs_update = True
    if needs_update and check_mode:
        if self.pool_detail['volumeGroupData']['diskPoolData']['poolUtilizationWarningThreshold'] != self.ddp_warning_threshold_pct:
            try:
                rc, update = self.request('storage-systems/%s/storage-pools/%s' % (self.ssid, self.pool_detail['id']), method='POST', data={'id': self.pool_detail['id'], 'poolThreshold': {'thresholdType': 'warning', 'value': self.ddp_warning_threshold_pct}})
            except Exception as error:
                self.module.fail_json(msg='Failed to update DDP warning alert threshold! Pool [%s]. Array [%s]. Error [%s]' % (self.name, self.ssid, to_native(error)))
        if self.pool_detail['volumeGroupData']['diskPoolData']['poolUtilizationCriticalThreshold'] != self.ddp_critical_threshold_pct:
            try:
                rc, update = self.request('storage-systems/%s/storage-pools/%s' % (self.ssid, self.pool_detail['id']), method='POST', data={'id': self.pool_detail['id'], 'poolThreshold': {'thresholdType': 'critical', 'value': self.ddp_critical_threshold_pct}})
            except Exception as error:
                self.module.fail_json(msg='Failed to update DDP critical alert threshold! Pool [%s]. Array [%s]. Error [%s]' % (self.name, self.ssid, to_native(error)))
    return needs_update