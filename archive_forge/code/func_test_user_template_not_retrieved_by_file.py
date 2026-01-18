import collections
import json
import os
from unittest import mock
import uuid
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import template_format
from heat.common import urlfetch
from heat.engine import attributes
from heat.engine import environment
from heat.engine import properties
from heat.engine import resource
from heat.engine import resources
from heat.engine.resources import template_resource
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import support
from heat.engine import template
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_user_template_not_retrieved_by_file(self):
    env = environment.Environment()
    test_templ_name = 'file:///etc/heatr/flippy.yaml'
    env.load({'resource_registry': {'Test::Flippy': test_templ_name}})
    stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template, env=env), stack_id=str(uuid.uuid4()))
    definition = rsrc_defn.ResourceDefinition('test_t_res', 'Test::Flippy')
    temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
    self.assertRaises(exception.StackValidationFailed, temp_res.validate)