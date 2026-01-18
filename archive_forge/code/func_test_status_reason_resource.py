from unittest import mock
import fixtures
from heat.common import exception
from heat.common.i18n import _
from heat.tests import common
def test_status_reason_resource(self):
    reason = 'Resource CREATE failed: ValueError: resources.oops: Test Resource failed oops'
    exc = exception.ResourceFailure(reason, None, action='CREATE')
    self.assertEqual('ValueError', exc.error)
    self.assertEqual(['resources', 'oops'], exc.path)
    self.assertEqual('Test Resource failed oops', exc.error_message)