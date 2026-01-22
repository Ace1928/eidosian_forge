from unittest import mock
import fixtures
from heat.common import exception
from heat.common.i18n import _
from heat.tests import common
class TestResourceFailure(common.HeatTestCase):

    def test_status_reason_resource(self):
        reason = 'Resource CREATE failed: ValueError: resources.oops: Test Resource failed oops'
        exc = exception.ResourceFailure(reason, None, action='CREATE')
        self.assertEqual('ValueError', exc.error)
        self.assertEqual(['resources', 'oops'], exc.path)
        self.assertEqual('Test Resource failed oops', exc.error_message)

    def test_status_reason_general(self):
        reason = 'something strange happened'
        exc = exception.ResourceFailure(reason, None, action='CREATE')
        self.assertEqual('', exc.error)
        self.assertEqual([], exc.path)
        self.assertEqual('something strange happened', exc.error_message)

    def test_status_reason_general_res(self):
        res = mock.Mock()
        res.name = 'fred'
        res.stack.t.get_section_name.return_value = 'Resources'
        reason = 'something strange happened'
        exc = exception.ResourceFailure(reason, res, action='CREATE')
        self.assertEqual('', exc.error)
        self.assertEqual(['Resources', 'fred'], exc.path)
        self.assertEqual('something strange happened', exc.error_message)

    def test_std_exception(self):
        base_exc = ValueError('sorry mom')
        exc = exception.ResourceFailure(base_exc, None, action='UPDATE')
        self.assertEqual('ValueError', exc.error)
        self.assertEqual([], exc.path)
        self.assertEqual('sorry mom', exc.error_message)

    def test_std_exception_with_resource(self):
        base_exc = ValueError('sorry mom')
        res = mock.Mock()
        res.name = 'fred'
        res.stack.t.get_section_name.return_value = 'Resources'
        exc = exception.ResourceFailure(base_exc, res, action='UPDATE')
        self.assertEqual('ValueError', exc.error)
        self.assertEqual(['Resources', 'fred'], exc.path)
        self.assertEqual('sorry mom', exc.error_message)

    def test_heat_exception(self):
        base_exc = ValueError('sorry mom')
        heat_exc = exception.ResourceFailure(base_exc, None, action='UPDATE')
        exc = exception.ResourceFailure(heat_exc, None, action='UPDATE')
        self.assertEqual('ValueError', exc.error)
        self.assertEqual([], exc.path)
        self.assertEqual('sorry mom', exc.error_message)

    def test_nested_exceptions(self):
        res = mock.Mock()
        res.name = 'frodo'
        res.stack.t.get_section_name.return_value = 'Resources'
        reason = 'Resource UPDATE failed: ValueError: resources.oops: Test Resource failed oops'
        base_exc = exception.ResourceFailure(reason, res, action='UPDATE')
        exc = exception.ResourceFailure(base_exc, res, action='UPDATE')
        self.assertEqual(['Resources', 'frodo', 'resources', 'oops'], exc.path)
        self.assertEqual('ValueError', exc.error)
        self.assertEqual('Test Resource failed oops', exc.error_message)