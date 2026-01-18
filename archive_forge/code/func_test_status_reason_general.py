from unittest import mock
import fixtures
from heat.common import exception
from heat.common.i18n import _
from heat.tests import common
def test_status_reason_general(self):
    reason = 'something strange happened'
    exc = exception.ResourceFailure(reason, None, action='CREATE')
    self.assertEqual('', exc.error)
    self.assertEqual([], exc.path)
    self.assertEqual('something strange happened', exc.error_message)