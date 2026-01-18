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
def test_system_template_retrieve_fail(self):
    g_env = resources.global_env()
    test_templ_name = 'file:///etc/heatr/frodo.yaml'
    g_env.load({'resource_registry': {'Test::Frodo': test_templ_name}})
    stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template), stack_id=str(uuid.uuid4()))
    mock_get = self.patchobject(urlfetch, 'get', side_effect=urlfetch.URLFetchError(_('Failed to retrieve template')))
    definition = rsrc_defn.ResourceDefinition('test_t_res', 'Test::Frodo')
    temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
    self.assertRaises(exception.StackValidationFailed, temp_res.validate)
    mock_get.assert_called_once_with(test_templ_name, allowed_schemes=('http', 'https', 'file'))