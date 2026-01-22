from unittest import mock
from heat.common import exception
from heat.engine.resources.openstack.heat import structured_config as sc
from heat.engine import rsrc_defn
from heat.engine import software_config_io as swc_io
from heat.engine import stack as parser
from heat.engine import template
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
class StructuredDeploymentParseMethodsTest(common.HeatTestCase):

    def test_get_key_args(self):
        snippet = {'get_input': 'bar'}
        input_key = 'get_input'
        expected = 'bar'
        result = sc.StructuredDeployment.get_input_key_arg(snippet, input_key)
        self.assertEqual(expected, result)

    def test_get_key_args_long_snippet(self):
        snippet = {'get_input': 'bar', 'second': 'foo'}
        input_key = 'get_input'
        result = sc.StructuredDeployment.get_input_key_arg(snippet, input_key)
        self.assertFalse(result)

    def test_get_key_args_unknown_input_key(self):
        snippet = {'get_input': 'bar'}
        input_key = 'input'
        result = sc.StructuredDeployment.get_input_key_arg(snippet, input_key)
        self.assertFalse(result)

    def test_get_key_args_wrong_args(self):
        snippet = {'get_input': None}
        input_key = 'get_input'
        result = sc.StructuredDeployment.get_input_key_arg(snippet, input_key)
        self.assertFalse(result)

    def test_get_input_key_value(self):
        inputs = {'bar': 'baz', 'foo': 'foo2'}
        res = sc.StructuredDeployment.get_input_key_value('bar', inputs, False)
        expected = 'baz'
        self.assertEqual(expected, res)

    def test_get_input_key_value_raise_exception(self):
        inputs = {'bar': 'baz', 'foo': 'foo2'}
        self.assertRaises(exception.UserParameterMissing, sc.StructuredDeployment.get_input_key_value, 'barz', inputs, 'STRICT')

    def test_get_input_key_value_get_none(self):
        inputs = {'bar': 'baz', 'foo': 'foo2'}
        res = sc.StructuredDeployment.get_input_key_value('brz', inputs, False)
        self.assertIsNone(res)