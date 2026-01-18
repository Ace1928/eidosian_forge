from unittest import mock
from unittest.mock import patch
import uuid
import testtools
from troveclient.v1 import backups
def test_schedule_show(self):
    instance = mock.Mock(id='the_uuid')
    backup_name = 'myback'
    cron_triggers = mock.Mock()
    cron_triggers.get = mock.Mock(return_value=mock.Mock(name=backup_name, workflow_input='{"name": "%s", "instance": "%s"}' % (backup_name, instance.id)))
    mistral_client = mock.Mock(cron_triggers=cron_triggers)
    sched = self.backups.schedule_show('dummy', mistral_client)
    self.assertEqual(backup_name, sched.name)
    self.assertEqual(instance.id, sched.instance)