import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import linear_flow
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
class FlowFromDetailTestCase(test.TestCase):

    def test_no_meta(self):
        _lb, flow_detail = p_utils.temporary_flow_detail()
        self.assertEqual({}, flow_detail.meta)
        self.assertRaisesRegex(ValueError, '^Cannot .* no factory information saved.$', taskflow.engines.flow_from_detail, flow_detail)

    def test_no_factory_in_meta(self):
        _lb, flow_detail = p_utils.temporary_flow_detail()
        self.assertRaisesRegex(ValueError, '^Cannot .* no factory information saved.$', taskflow.engines.flow_from_detail, flow_detail)

    def test_no_importable_function(self):
        _lb, flow_detail = p_utils.temporary_flow_detail()
        flow_detail.meta = dict(factory=dict(name='you can not import me, i contain spaces'))
        self.assertRaisesRegex(ImportError, '^Could not import factory', taskflow.engines.flow_from_detail, flow_detail)

    def test_no_arg_factory(self):
        name = 'some.test.factory'
        _lb, flow_detail = p_utils.temporary_flow_detail()
        flow_detail.meta = dict(factory=dict(name=name))
        with mock.patch('oslo_utils.importutils.import_class', return_value=lambda: 'RESULT') as mock_import:
            result = taskflow.engines.flow_from_detail(flow_detail)
            mock_import.assert_called_once_with(name)
        self.assertEqual('RESULT', result)

    def test_factory_with_arg(self):
        name = 'some.test.factory'
        _lb, flow_detail = p_utils.temporary_flow_detail()
        flow_detail.meta = dict(factory=dict(name=name, args=['foo']))
        with mock.patch('oslo_utils.importutils.import_class', return_value=lambda x: 'RESULT %s' % x) as mock_import:
            result = taskflow.engines.flow_from_detail(flow_detail)
            mock_import.assert_called_once_with(name)
        self.assertEqual('RESULT foo', result)