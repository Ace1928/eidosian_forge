import collections
import copy
import datetime
import json
import logging
import time
from unittest import mock
import eventlet
import fixtures
from oslo_config import cfg
from heat.common import context
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.db import api as db_api
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine import node_data
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import service
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.engine import update
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import stack as stack_object
from heat.objects import stack_tag as stack_tag_object
from heat.objects import user_creds as ucreds_object
from heat.tests import common
from heat.tests import fakes
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
@mock.patch.object(resource_objects.Resource, 'get_all_by_stack')
def test_resource_get_db_fallback(self, gabs):
    tpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'A': {'Type': 'GenericResourceType'}}}
    self.stack = stack.Stack(self.ctx, 'test_stack', template.Template(tpl), status_reason='blarg')
    self.stack.store()
    tpl2 = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'A': {'Type': 'GenericResourceType'}, 'B': {'Type': 'GenericResourceType'}}}
    t2 = template.Template(tpl2)
    t2.store(self.ctx)
    db_resources = {'A': mock.MagicMock(), 'B': mock.MagicMock(current_template_id=t2.id), 'C': mock.MagicMock(current_template_id=t2.id)}
    db_resources['A'].name = 'A'
    db_resources['B'].name = 'B'
    db_resources['C'].name = 'C'
    gabs.return_value = db_resources
    self.assertEqual('A', self.stack.resource_get('A').name)
    self.assertEqual('B', self.stack.resource_get('B').name)
    self.assertIsNone(self.stack.resource_get('C'))
    self.assertIsNone(self.stack.resource_get('D'))