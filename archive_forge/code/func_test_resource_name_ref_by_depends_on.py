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
def test_resource_name_ref_by_depends_on(self):
    tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'AResource': {'Type': 'GenericResourceType'}, 'BResource': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': 'AResource'}, 'DependsOn': 'AResource'}}}
    self.stack = stack.Stack(self.ctx, 'resource_by_name_ref_stack', template.Template(tmpl))
    self.stack.store()
    self.stack.create()
    self.assertEqual((stack.Stack.CREATE, stack.Stack.COMPLETE), self.stack.state)
    self.assertIn('AResource', self.stack)
    self.assertIn('BResource', self.stack)
    rsrc = self.stack['AResource']
    rsrc.resource_id_set('aaaa')
    b_rsrc = self.stack['BResource']
    b_rsrc.resource_id_set('bbbb')
    b_foo_ref = b_rsrc.properties.get('Foo')
    for action, status in ((rsrc.INIT, rsrc.COMPLETE), (rsrc.CREATE, rsrc.IN_PROGRESS), (rsrc.CREATE, rsrc.COMPLETE), (rsrc.RESUME, rsrc.IN_PROGRESS), (rsrc.RESUME, rsrc.COMPLETE), (rsrc.UPDATE, rsrc.IN_PROGRESS), (rsrc.UPDATE, rsrc.COMPLETE)):
        rsrc.state_set(action, status)
        ref_rsrc = self.stack.resource_by_refid(b_foo_ref)
        self.assertEqual(rsrc, ref_rsrc)
        self.assertIn(b_rsrc.name, ref_rsrc.required_by())