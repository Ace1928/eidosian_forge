import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_two_event_triggers_for_one_wf(self):
    self.event_trigger_create('trigger1', self.wf_id, 'dummy_exchange', 'dummy_topic', 'event.dummy', '{}')
    self.event_trigger_create('trigger2', self.wf_id, 'dummy_exchange', 'dummy_topic', 'dummy.event', '{}')
    triggers = self.mistral_admin('event-trigger-list')
    self.assertIn('trigger1', [tr['Name'] for tr in triggers])
    self.assertIn('trigger2', [tr['Name'] for tr in triggers])