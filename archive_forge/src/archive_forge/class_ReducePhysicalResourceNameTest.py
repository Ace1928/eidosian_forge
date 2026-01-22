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
class ReducePhysicalResourceNameTest(common.HeatTestCase):
    scenarios = [('one', dict(limit=10, original='one', reduced='one')), ('limit_plus_one', dict(will_reduce=True, limit=10, original='onetwothree', reduced='on-wothree')), ('limit_exact', dict(limit=11, original='onetwothree', reduced='onetwothree')), ('limit_minus_one', dict(limit=12, original='onetwothree', reduced='onetwothree')), ('limit_four', dict(will_reduce=True, limit=4, original='onetwothree', reduced='on-e')), ('limit_three', dict(will_raise=ValueError, limit=3, original='onetwothree')), ('three_nested_stacks', dict(will_reduce=True, limit=63, original='ElasticSearch-MasterCluster-ccicxsm25ug6-MasterSvr1-men65r4t53hh-MasterServer-gxpc3wqxy4el', reduced='El-icxsm25ug6-MasterSvr1-men65r4t53hh-MasterServer-gxpc3wqxy4el')), ('big_names', dict(will_reduce=True, limit=63, original='MyReallyQuiteVeryLongStackName-MyExtraordinarilyLongResourceName-ccicxsm25ug6', reduced='My-LongStackName-MyExtraordinarilyLongResourceName-ccicxsm25ug6'))]
    will_raise = None
    will_reduce = False

    def test_reduce(self):
        if self.will_raise:
            self.assertRaises(self.will_raise, resource.Resource.reduce_physical_resource_name, self.original, self.limit)
        else:
            reduced = resource.Resource.reduce_physical_resource_name(self.original, self.limit)
            self.assertEqual(self.reduced, reduced)
            if self.will_reduce:
                self.assertEqual(self.limit, len(reduced))
            else:
                self.assertEqual(self.original, reduced)