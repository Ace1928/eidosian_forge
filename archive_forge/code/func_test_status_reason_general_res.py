from unittest import mock
import fixtures
from heat.common import exception
from heat.common.i18n import _
from heat.tests import common
def test_status_reason_general_res(self):
    res = mock.Mock()
    res.name = 'fred'
    res.stack.t.get_section_name.return_value = 'Resources'
    reason = 'something strange happened'
    exc = exception.ResourceFailure(reason, res, action='CREATE')
    self.assertEqual('', exc.error)
    self.assertEqual(['Resources', 'fred'], exc.path)
    self.assertEqual('something strange happened', exc.error_message)