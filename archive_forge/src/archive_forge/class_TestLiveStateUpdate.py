import collections
import datetime
import itertools
import json
import os
import sys
from unittest import mock
import uuid
import eventlet
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import short_id
from heat.common import timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import attributes
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import clients
from heat.engine import constraints
from heat.engine import dependencies
from heat.engine import environment
from heat.engine import node_data
from heat.engine import plugin_manager
from heat.engine import properties
from heat.engine import resource
from heat.engine import resources
from heat.engine.resources.openstack.heat import none_resource
from heat.engine.resources.openstack.heat import test_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import support
from heat.engine import template
from heat.engine import translation
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_object
from heat.objects import resource_properties_data as rpd_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
import neutronclient.common.exceptions as neutron_exp
class TestLiveStateUpdate(common.HeatTestCase):
    scenarios = [('update_all_args', dict(live_state={'Foo': 'abb', 'FooInt': 2}, updated_props={'Foo': 'bca', 'FooInt': 3}, expected_error=False, resource_id='1234', expected={'Foo': 'bca', 'FooInt': 3})), ('update_some_args', dict(live_state={'Foo': 'bca'}, updated_props={'Foo': 'bca', 'FooInt': 3}, expected_error=False, resource_id='1234', expected={'Foo': 'bca', 'FooInt': 3})), ('live_state_some_error', dict(live_state={'Foo': 'bca'}, updated_props={'Foo': 'bca', 'FooInt': 3}, expected_error=False, resource_id='1234', expected={'Foo': 'bca', 'FooInt': 3})), ('entity_not_found', dict(live_state=exception.EntityNotFound(entity='resource', name='test'), updated_props={'Foo': 'bca'}, expected_error=True, resource_id='1234', expected=resource.UpdateReplace)), ('live_state_not_found_id', dict(live_state=exception.EntityNotFound(entity='resource', name='test'), updated_props={'Foo': 'bca'}, expected_error=True, resource_id=None, expected=resource.UpdateReplace))]

    def setUp(self):
        super(TestLiveStateUpdate, self).setUp()
        self.env = environment.Environment()
        self.env.load({u'resource_registry': {u'OS::Test::GenericResource': u'GenericResourceType', u'OS::Test::ResourceWithCustomConstraint': u'ResourceWithCustomConstraint'}})
        self.stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template, env=self.env), stack_id=str(uuid.uuid4()))

    def _prepare_resource_live_state(self):
        tmpl = rsrc_defn.ResourceDefinition('test_resource', 'ResourceWithPropsType', {'Foo': 'abc', 'FooInt': 2})
        res = generic_rsrc.ResourceWithProps('test_resource', tmpl, self.stack)
        for prop in res.properties.props.values():
            prop.schema.update_allowed = True
        res.update_allowed_properties = ('Foo', 'FooInt')
        scheduler.TaskRunner(res.create)()
        self.assertEqual((res.CREATE, res.COMPLETE), res.state)
        return res

    def _clean_tests_after_resource_live_state(self, res):
        """Revert changes for correct work of other tests.

        Need to revert changes of resource properties schema for correct work
        of other tests.
        """
        res.update_allowed_properties = []
        res.update_allowed_set = []
        for prop in res.properties.props.values():
            prop.schema.update_allowed = False

    def test_update_resource_live_state(self):
        res = self._prepare_resource_live_state()
        res.resource_id = self.resource_id
        cfg.CONF.set_override('observe_on_update', True)
        utmpl = rsrc_defn.ResourceDefinition('test_resource', 'ResourceWithPropsType', self.updated_props)
        if not self.expected_error:
            self.patchobject(res, 'get_live_state', return_value=self.live_state)
            scheduler.TaskRunner(res.update, utmpl)()
            self.assertEqual((res.UPDATE, res.COMPLETE), res.state)
            self.assertEqual(self.expected, res.properties.data)
        else:
            self.patchobject(res, 'get_live_state', side_effect=[self.live_state])
            self.assertRaises(self.expected, scheduler.TaskRunner(res.update, utmpl))
        self._clean_tests_after_resource_live_state(res)

    def test_get_live_resource_data_success(self):
        res = self._prepare_resource_live_state()
        res.resource_id = self.resource_id
        res._show_resource = mock.MagicMock(return_value={'a': 'b'})
        self.assertEqual({'a': 'b'}, res.get_live_resource_data())
        self._clean_tests_after_resource_live_state(res)

    def test_get_live_resource_data_not_found(self):
        res = self._prepare_resource_live_state()
        res.default_client_name = 'foo'
        res.resource_id = self.resource_id
        res._show_resource = mock.MagicMock(side_effect=[exception.NotFound()])
        res.client_plugin = mock.MagicMock()
        res.client_plugin().is_not_found = mock.MagicMock(return_value=True)
        ex = self.assertRaises(exception.EntityNotFound, res.get_live_resource_data)
        self.assertEqual('The Resource (test_resource) could not be found.', str(ex))
        self._clean_tests_after_resource_live_state(res)

    def test_parse_live_resource_data(self):
        res = self._prepare_resource_live_state()
        res.update_allowed_props = mock.Mock(return_value=['Foo', 'Bar'])
        resource_data = {'Foo': 'brave new data', 'Something not so good': 'for all of us'}
        res._update_allowed_properties = ['Foo', 'Bar']
        result = res.parse_live_resource_data(res.properties, resource_data)
        self.assertEqual({'Foo': 'brave new data'}, result)
        self._clean_tests_after_resource_live_state(res)

    def test_get_live_resource_data_not_found_no_default_client_name(self):

        class MyException(Exception):
            pass
        res = self._prepare_resource_live_state()
        res.default_client_name = None
        res.resource_id = self.resource_id
        res._show_resource = mock.MagicMock(side_effect=[MyException])
        res.client_plugin = mock.MagicMock()
        res.client_plugin().is_not_found = mock.MagicMock(return_value=True)
        self.assertRaises(MyException, res.get_live_resource_data)
        self._clean_tests_after_resource_live_state(res)

    def test_get_live_resource_data_other_error(self):
        res = self._prepare_resource_live_state()
        res.resource_id = self.resource_id
        res._show_resource = mock.MagicMock(side_effect=[exception.Forbidden()])
        res.client_plugin = mock.MagicMock()
        res.client_plugin().is_not_found = mock.MagicMock(return_value=False)
        self.assertRaises(exception.Forbidden, res.get_live_resource_data)
        self._clean_tests_after_resource_live_state(res)