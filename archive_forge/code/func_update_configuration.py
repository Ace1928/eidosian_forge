from __future__ import absolute_import, division, print_function
import json
import logging
from pprint import pformat
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
def update_configuration(self):
    config = self.get_configuration()
    update = False
    body = dict()
    if self.asup:
        body = dict(asupEnabled=True)
        if not config['asupEnabled']:
            update = True
        if (config['onDemandEnabled'] and config['remoteDiagsEnabled']) != self.active:
            update = True
            body.update(dict(onDemandEnabled=self.active, remoteDiagsEnabled=self.active))
        self.days.sort()
        config['schedule']['daysOfWeek'].sort()
        body['schedule'] = dict(daysOfWeek=self.days, dailyMinTime=self.start, dailyMaxTime=self.end, weeklyMinTime=self.start, weeklyMaxTime=self.end)
        if self.days != config['schedule']['daysOfWeek']:
            update = True
        if self.start != config['schedule']['dailyMinTime'] or self.start != config['schedule']['weeklyMinTime']:
            update = True
        elif self.end != config['schedule']['dailyMaxTime'] or self.end != config['schedule']['weeklyMaxTime']:
            update = True
    elif config['asupEnabled']:
        body = dict(asupEnabled=False)
        update = True
    self._logger.info(pformat(body))
    if update and (not self.check_mode):
        try:
            rc, result = request(self.url + 'device-asup', method='POST', data=json.dumps(body), headers=HEADERS, **self.creds)
        except Exception as err:
            self.module.fail_json(msg='We failed to set the storage-system name! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
    return update