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
def test_resolve_attributes_stuff_base_attribute(self):
    stack = self.create_resource_for_attributes_tests()
    res = stack['res']

    class MyException(Exception):
        pass
    with mock.patch.object(res, '_show_resource') as show_attr:
        self.assertIsNone(res.FnGetAtt('show'))
        res.resource_id = mock.Mock()
        res.default_client_name = 'foo'
        show_attr.return_value = 'my attr'
        self.assertEqual('my attr', res.FnGetAtt('show'))
        self.assertEqual(1, show_attr.call_count)
        res.attributes.reset_resolved_values()
        show_attr.side_effect = [MyException]
        self.assertRaises(MyException, res.FnGetAtt, 'show')
        self.assertEqual(2, show_attr.call_count)