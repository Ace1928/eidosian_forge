import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from heat.common import crypt
from heat.common import exception
from heat.common import template_format
from heat.db import api as db_api
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import service
from heat.engine import service_software_config
from heat.engine import software_config_io as swc_io
from heat.objects import resource as resource_objects
from heat.objects import software_config as software_config_object
from heat.objects import software_deployment as software_deployment_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
class SoftwareConfigIOSchemaTest(common.HeatTestCase):

    def test_input_config_empty(self):
        name = 'foo'
        inp = swc_io.InputConfig(name=name)
        self.assertIsNone(inp.default())
        self.assertIs(False, inp.replace_on_change())
        self.assertEqual(name, inp.name())
        self.assertEqual({'name': name, 'type': 'String'}, inp.as_dict())
        self.assertEqual((name, None), inp.input_data())

    def test_input_config(self):
        name = 'bar'
        inp = swc_io.InputConfig(name=name, description='test', type='Number', default=0, replace_on_change=True)
        self.assertEqual(0, inp.default())
        self.assertIs(True, inp.replace_on_change())
        self.assertEqual(name, inp.name())
        self.assertEqual({'name': name, 'type': 'Number', 'description': 'test', 'default': 0, 'replace_on_change': True}, inp.as_dict())
        self.assertEqual((name, None), inp.input_data())

    def test_input_config_value(self):
        name = 'baz'
        inp = swc_io.InputConfig(name=name, type='Number', default=0, value=42)
        self.assertEqual(0, inp.default())
        self.assertIs(False, inp.replace_on_change())
        self.assertEqual(name, inp.name())
        self.assertEqual({'name': name, 'type': 'Number', 'default': 0, 'value': 42}, inp.as_dict())
        self.assertEqual((name, 42), inp.input_data())

    def test_input_config_no_name(self):
        self.assertRaises(ValueError, swc_io.InputConfig, type='String')

    def test_input_config_extra_key(self):
        self.assertRaises(ValueError, swc_io.InputConfig, name='test', bogus='wat')

    def test_input_types(self):
        swc_io.InputConfig(name='str', type='String').as_dict()
        swc_io.InputConfig(name='num', type='Number').as_dict()
        swc_io.InputConfig(name='list', type='CommaDelimitedList').as_dict()
        swc_io.InputConfig(name='json', type='Json').as_dict()
        swc_io.InputConfig(name='bool', type='Boolean').as_dict()
        self.assertRaises(ValueError, swc_io.InputConfig, name='bogus', type='BogusType')

    def test_output_config_empty(self):
        name = 'foo'
        outp = swc_io.OutputConfig(name=name)
        self.assertEqual(name, outp.name())
        self.assertEqual({'name': name, 'type': 'String', 'error_output': False}, outp.as_dict())

    def test_output_config(self):
        name = 'bar'
        outp = swc_io.OutputConfig(name=name, description='test', type='Json', error_output=True)
        self.assertEqual(name, outp.name())
        self.assertIs(True, outp.error_output())
        self.assertEqual({'name': name, 'type': 'Json', 'description': 'test', 'error_output': True}, outp.as_dict())

    def test_output_config_no_name(self):
        self.assertRaises(ValueError, swc_io.OutputConfig, type='String')

    def test_output_config_extra_key(self):
        self.assertRaises(ValueError, swc_io.OutputConfig, name='test', bogus='wat')

    def test_output_types(self):
        swc_io.OutputConfig(name='str', type='String').as_dict()
        swc_io.OutputConfig(name='num', type='Number').as_dict()
        swc_io.OutputConfig(name='list', type='CommaDelimitedList').as_dict()
        swc_io.OutputConfig(name='json', type='Json').as_dict()
        swc_io.OutputConfig(name='bool', type='Boolean').as_dict()
        self.assertRaises(ValueError, swc_io.OutputConfig, name='bogus', type='BogusType')

    def test_check_io_schema_empty_list(self):
        swc_io.check_io_schema_list([])

    def test_check_io_schema_string(self):
        self.assertRaises(TypeError, swc_io.check_io_schema_list, '')

    def test_check_io_schema_dict(self):
        self.assertRaises(TypeError, swc_io.check_io_schema_list, {})

    def test_check_io_schema_list_dict(self):
        swc_io.check_io_schema_list([{'name': 'foo'}])

    def test_check_io_schema_list_string(self):
        self.assertRaises(TypeError, swc_io.check_io_schema_list, ['foo'])

    def test_check_io_schema_list_list(self):
        self.assertRaises(TypeError, swc_io.check_io_schema_list, [['foo']])

    def test_check_io_schema_list_none(self):
        self.assertRaises(TypeError, swc_io.check_io_schema_list, [None])

    def test_check_io_schema_list_mixed(self):
        self.assertRaises(TypeError, swc_io.check_io_schema_list, [{'name': 'foo'}, ('name', 'bar')])

    def test_input_config_value_json_default(self):
        name = 'baz'
        inp = swc_io.InputConfig(name=name, type='Json', default={'a': 1}, value=42)
        self.assertEqual({'a': 1}, inp.default())

    def test_input_config_value_default_coerce(self):
        name = 'baz'
        inp = swc_io.InputConfig(name=name, type='Number', default='0')
        self.assertEqual(0, inp.default())

    def test_input_config_value_ignore_string(self):
        name = 'baz'
        inp = swc_io.InputConfig(name=name, type='Number', default='')
        self.assertEqual({'type': 'Number', 'name': 'baz', 'default': ''}, inp.as_dict())