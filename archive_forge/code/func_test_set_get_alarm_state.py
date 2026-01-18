import os
from oslo_utils import uuidutils
import requests
import requests.auth
from tempest.lib import exceptions
from aodhclient.tests.functional import base
def test_set_get_alarm_state(self):
    result = self.aodh('alarm', params='create --type event --name alarm_state_test --query "traits.project_id=789;traits.resource_id=012"')
    alarm = self.details_multiple(result)[0]
    alarm_id = alarm['alarm_id']
    result = self.aodh('alarm', params='show %s' % alarm_id)
    alarm_show = self.details_multiple(result)[0]
    self.assertEqual('insufficient data', alarm_show['state'])
    result = self.aodh('alarm', params='state get %s' % alarm_id)
    state_get = self.details_multiple(result)[0]
    self.assertEqual('insufficient data', state_get['state'])
    self.aodh('alarm', params='state set --state ok  %s' % alarm_id)
    result = self.aodh('alarm', params='state get %s' % alarm_id)
    state_get = self.details_multiple(result)[0]
    self.assertEqual('ok', state_get['state'])
    self.aodh('alarm', params='delete %s' % alarm_id)