from unittest import mock
from unittest.mock import patch
import uuid
import testtools
from troveclient.v1 import backups
def test_schedule_delete(self):
    cron_triggers = mock.Mock()
    cron_triggers.delete = mock.Mock()
    mistral_client = mock.Mock(cron_triggers=cron_triggers)
    self.backups.schedule_delete('dummy', mistral_client)
    cron_triggers.delete.assert_called()