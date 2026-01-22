import datetime as dt
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import api
from heat.engine.cfn import parameters as cfn_param
from heat.engine import event
from heat.engine import parent_rsrc
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import event as event_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
class ParseStatusTest(common.HeatTestCase):
    scenarios = [('single_bogus', dict(inputs='bogus status', expected=(set(), set()))), ('list_bogus', dict(inputs=['foo', 'bar'], expected=(set(), set()))), ('single_partial', dict(inputs='COMPLETE', expected=(set(), set(['COMPLETE'])))), ('multi_partial', dict(inputs=['FAILED', 'COMPLETE'], expected=(set(), set(['FAILED', 'COMPLETE'])))), ('multi_partial_dup', dict(inputs=['FAILED', 'FAILED'], expected=(set(), set(['FAILED'])))), ('single_full', dict(inputs=['DELETE_FAILED'], expected=(set(['DELETE']), set(['FAILED'])))), ('multi_full', dict(inputs=['DELETE_FAILED', 'CREATE_COMPLETE'], expected=(set(['CREATE', 'DELETE']), set(['COMPLETE', 'FAILED'])))), ('mix_bogus_partial', dict(inputs=['delete_failed', 'COMPLETE'], expected=(set(), set(['COMPLETE'])))), ('mix_bogus_full', dict(inputs=['delete_failed', 'action_COMPLETE'], expected=(set(['action']), set(['COMPLETE'])))), ('mix_bogus_full_incomplete', dict(inputs=['delete_failed', '_COMPLETE'], expected=(set(), set(['COMPLETE'])))), ('mix_partial_full', dict(inputs=['FAILED', 'b_COMPLETE'], expected=(set(['b']), set(['COMPLETE', 'FAILED'])))), ('mix_full_dup', dict(inputs=['a_FAILED', 'a_COMPLETE'], expected=(set(['a']), set(['COMPLETE', 'FAILED'])))), ('mix_full_dup_2', dict(inputs=['a_FAILED', 'b_FAILED'], expected=(set(['a', 'b']), set(['FAILED']))))]

    def test_stack_parse_status(self):
        actual = api._parse_object_status(self.inputs)
        self.assertEqual(self.expected, actual)