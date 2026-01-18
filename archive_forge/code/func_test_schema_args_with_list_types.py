import io
import sys
from unittest import mock
from oslo_utils import encodeutils
from requests import Response
import testtools
from glanceclient.common import utils
def test_schema_args_with_list_types(self):

    def schema_getter(_type='string', enum=False):
        prop = {'type': ['null', _type], 'description': 'Test schema'}
        prop_readonly = {'type': ['null', _type], 'readOnly': True, 'description': 'Test schema read-only'}
        if enum:
            prop['enum'] = [None, 'opt-1', 'opt-2']
            prop_readonly['enum'] = [None, 'opt-ro-1', 'opt-ro-2']

        def actual_getter():
            return {'additionalProperties': False, 'required': ['name'], 'name': 'test_schema', 'properties': {'test': prop, 'readonly-test': prop_readonly}}
        return actual_getter

    def dummy_func():
        pass
    decorated = utils.schema_args(schema_getter())(dummy_func)
    self.assertEqual(len(decorated.__dict__['arguments']), 1)
    arg, opts = decorated.__dict__['arguments'][0]
    self.assertIn('--test', arg)
    self.assertEqual(encodeutils.safe_decode, opts['type'])
    decorated = utils.schema_args(schema_getter('integer'))(dummy_func)
    arg, opts = decorated.__dict__['arguments'][0]
    self.assertIn('--test', arg)
    self.assertEqual(int, opts['type'])
    decorated = utils.schema_args(schema_getter(enum=True))(dummy_func)
    arg, opts = decorated.__dict__['arguments'][0]
    self.assertIn('--test', arg)
    self.assertEqual(encodeutils.safe_decode, opts['type'])
    self.assertIn('None, opt-1, opt-2', opts['help'])
    decorated = utils.schema_args(schema_getter('boolean'))(dummy_func)
    arg, opts = decorated.__dict__['arguments'][0]
    type_function = opts['type']
    self.assertEqual(type_function('False'), False)
    self.assertEqual(type_function('True'), True)
    self.assertRaises(ValueError, type_function, 'foo')