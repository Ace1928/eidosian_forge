import copy
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.db import api as db_api
from heat.engine.clients.os.keystone import fake_keystoneclient
from heat.engine import environment
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import service
from heat.engine import stack
from heat.engine import support
from heat.engine import template
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_update_tags(self):
    tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Description': 'ATemplate', 'Resources': {'AResource': {'Type': 'GenericResourceType'}}}
    self.stack = stack.Stack(self.ctx, 'update_test_stack', template.Template(tmpl), tags=['tag1', 'tag2'])
    self.stack.store()
    self.stack.create()
    self.assertEqual((stack.Stack.CREATE, stack.Stack.COMPLETE), self.stack.state)
    self.assertEqual(['tag1', 'tag2'], self.stack.tags)
    updated_stack = stack.Stack(self.ctx, 'updated_stack', template.Template(tmpl), tags=['tag3', 'tag4'])
    self.stack.update(updated_stack)
    self.assertEqual((stack.Stack.UPDATE, stack.Stack.COMPLETE), self.stack.state)
    self.assertEqual(['tag3', 'tag4'], self.stack.tags)