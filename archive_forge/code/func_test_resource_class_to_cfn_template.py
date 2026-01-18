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
def test_resource_class_to_cfn_template(self):

    class TestResource(resource.Resource):
        list_schema = {'wont_show_up': {'Type': 'Number'}}
        map_schema = {'will_show_up': {'Type': 'Integer'}}
        properties_schema = {'name': {'Type': 'String'}, 'bool': {'Type': 'Boolean'}, 'implemented': {'Type': 'String', 'Implemented': True, 'AllowedPattern': '.*', 'MaxLength': 7, 'MinLength': 2, 'Required': True}, 'not_implemented': {'Type': 'String', 'Implemented': False}, 'number': {'Type': 'Number', 'MaxValue': 77, 'MinValue': 41, 'Default': 42}, 'list': {'Type': 'List', 'Schema': {'Type': 'Map', 'Schema': list_schema}}, 'map': {'Type': 'Map', 'Schema': map_schema}, 'hidden': properties.Schema(properties.Schema.STRING, support_status=support.SupportStatus(status=support.HIDDEN))}
        attributes_schema = {'output1': attributes.Schema('output1_desc'), 'output2': attributes.Schema('output2_desc')}
    expected_template = {'HeatTemplateFormatVersion': '2012-12-12', 'Description': 'Initial template of TestResource', 'Parameters': {'name': {'Type': 'String'}, 'bool': {'Type': 'Boolean', 'AllowedValues': ['True', 'true', 'False', 'false']}, 'implemented': {'Type': 'String', 'AllowedPattern': '.*', 'MaxLength': 7, 'MinLength': 2}, 'number': {'Type': 'Number', 'MaxValue': 77, 'MinValue': 41, 'Default': 42}, 'list': {'Type': 'CommaDelimitedList'}, 'map': {'Type': 'Json'}}, 'Resources': {'TestResource': {'Type': 'Test::Resource::resource', 'Properties': {'name': {'Ref': 'name'}, 'bool': {'Ref': 'bool'}, 'implemented': {'Ref': 'implemented'}, 'number': {'Ref': 'number'}, 'list': {'Fn::Split': [',', {'Ref': 'list'}]}, 'map': {'Ref': 'map'}}}}, 'Outputs': {'output1': {'Description': 'output1_desc', 'Value': {'Fn::GetAtt': ['TestResource', 'output1']}}, 'output2': {'Description': 'output2_desc', 'Value': {'Fn::GetAtt': ['TestResource', 'output2']}}, 'show': {'Description': u'Detailed information about resource.', 'Value': {'Fn::GetAtt': ['TestResource', 'show']}}, 'OS::stack_id': {'Value': {'Ref': 'TestResource'}}}}
    self.assertEqual(expected_template, TestResource.resource_to_template('Test::Resource::resource'))