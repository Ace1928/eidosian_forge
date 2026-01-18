import copy
from unittest import mock
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine.cfn import functions as cfn_functions
from heat.engine.cfn import parameters as cfn_param
from heat.engine import conditions
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import functions as hot_functions
from heat.engine.hot import parameters as hot_param
from heat.engine.hot import template as hot_template
from heat.engine import resource
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_yaql_as_condition(self):
    hot_tpl = template_format.parse("\n        heat_template_version: pike\n        parameters:\n          ServiceNames:\n            type: comma_delimited_list\n            default: ['neutron', 'heat']\n        ")
    snippet = {'yaql': {'expression': '$.data.service_names.contains("neutron")', 'data': {'service_names': {'get_param': 'ServiceNames'}}}}
    tmpl = template.Template(hot_tpl)
    stack = parser.Stack(utils.dummy_context(), 'test_condition_yaql_true', tmpl)
    resolved = self.resolve_condition(snippet, tmpl, stack)
    self.assertTrue(resolved)
    tmpl = template.Template(hot_tpl, env=environment.Environment({'ServiceNames': ['nova_network', 'heat']}))
    stack = parser.Stack(utils.dummy_context(), 'test_condition_yaql_false', tmpl)
    resolved = self.resolve_condition(snippet, tmpl, stack)
    self.assertFalse(resolved)