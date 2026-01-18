from unittest import mock
from unittest.mock import patch
import uuid
import testtools
from troveclient.v1 import backups
def test_schedule_list(self):
    instance = mock.Mock(id='the_uuid')
    backup_name = 'wf2'
    test_input = [('wf1', 'foo'), (backup_name, instance.id)]
    cron_triggers = mock.Mock()
    cron_triggers.list = mock.Mock(return_value=[mock.Mock(workflow_input='{"name": "%s", "instance": "%s"}' % (name, inst), name=name) for name, inst in test_input])
    mistral_client = mock.Mock(cron_triggers=cron_triggers)
    sched_list = self.backups.schedule_list(instance, mistral_client)
    self.assertEqual(1, len(sched_list))
    the_sched = sched_list.pop()
    self.assertEqual(backup_name, the_sched.name)
    self.assertEqual(instance.id, the_sched.instance)