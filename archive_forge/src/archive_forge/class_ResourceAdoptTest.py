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
class ResourceAdoptTest(common.HeatTestCase):

    def test_adopt_resource_success(self):
        adopt_data = '{}'
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'foo': {'Type': 'GenericResourceType'}}})
        self.stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl, stack_id=str(uuid.uuid4()), adopt_stack_data=json.loads(adopt_data))
        res = self.stack['foo']
        res_data = {'status': 'COMPLETE', 'name': 'foo', 'resource_data': {}, 'metadata': {}, 'resource_id': 'test-res-id', 'action': 'CREATE', 'type': 'GenericResourceType'}
        adopt = scheduler.TaskRunner(res.adopt, res_data)
        adopt()
        self.assertEqual({}, res.metadata_get())
        self.assertEqual((res.ADOPT, res.COMPLETE), res.state)

    def test_adopt_with_resource_data_and_metadata(self):
        adopt_data = '{}'
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'foo': {'Type': 'GenericResourceType'}}})
        self.stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl, stack_id=str(uuid.uuid4()), adopt_stack_data=json.loads(adopt_data))
        res = self.stack['foo']
        res_data = {'status': 'COMPLETE', 'name': 'foo', 'resource_data': {'test-key': 'test-value'}, 'metadata': {'os_distro': 'test-distro'}, 'resource_id': 'test-res-id', 'action': 'CREATE', 'type': 'GenericResourceType'}
        adopt = scheduler.TaskRunner(res.adopt, res_data)
        adopt()
        self.assertEqual('test-value', resource_data_object.ResourceData.get_val(res, 'test-key'))
        self.assertEqual({'os_distro': 'test-distro'}, res.metadata_get())
        self.assertEqual((res.ADOPT, res.COMPLETE), res.state)

    def test_adopt_resource_missing(self):
        adopt_data = '{\n                        "action": "CREATE",\n                        "status": "COMPLETE",\n                        "name": "my-test-stack-name",\n                        "resources": {}\n                        }'
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'foo': {'Type': 'GenericResourceType'}}})
        self.stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl, stack_id=str(uuid.uuid4()), adopt_stack_data=json.loads(adopt_data))
        res = self.stack['foo']
        adopt = scheduler.TaskRunner(res.adopt, None)
        self.assertRaises(exception.ResourceFailure, adopt)
        expected = 'Exception: resources.foo: Resource ID was not provided.'
        self.assertEqual(expected, res.status_reason)