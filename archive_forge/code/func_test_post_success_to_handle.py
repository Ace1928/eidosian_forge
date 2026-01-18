import copy
import datetime
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from urllib import parse
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine import environment
from heat.engine import node_data
from heat.engine.resources.aws.cfn import wait_condition_handle as aws_wch
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.objects import resource as resource_objects
from heat.tests import common
from heat.tests import utils
def test_post_success_to_handle(self):
    self.stack = self.create_stack()
    self.m_gs.side_effect = [[], [], ['SUCCESS']]
    self.stack.create()
    rsrc = self.stack['WaitForTheHandle']
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    r = resource_objects.Resource.get_by_name_and_stack(self.stack.context, 'WaitHandle', self.stack.id)
    self.assertEqual('WaitHandle', r.name)
    self.assertEqual(3, self.m_gs.call_count)
    self.assertEqual(1, self.m_id.call_count)