from unittest import mock
from unittest.mock import patch
import uuid
import testtools
from troveclient.v1 import backups
def test_schedule_create(self):
    instance = mock.Mock()
    pattern = mock.Mock()
    name = 'myback'

    def make_cron_trigger(name, wf, workflow_input=None, pattern=None):
        return mock.Mock(name=name, pattern=pattern, workflow_input=workflow_input)
    cron_triggers = mock.Mock()
    cron_triggers.create = mock.Mock(side_effect=make_cron_trigger)
    mistral_client = mock.Mock(cron_triggers=cron_triggers)
    sched = self.backups.schedule_create(instance, pattern, name, mistral_client=mistral_client)
    self.assertEqual(pattern, sched.pattern)
    self.assertEqual(name, sched.name)
    self.assertEqual(instance.id, sched.instance)