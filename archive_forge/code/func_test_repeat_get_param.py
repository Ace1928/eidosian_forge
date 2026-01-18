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
def test_repeat_get_param(self):
    """Test repeat function with get_param function as an argument."""
    hot_tpl = template_format.parse("\n        heat_template_version: 2015-04-30\n        parameters:\n          param:\n            type: comma_delimited_list\n            default: 'a,b,c'\n        ")
    snippet = {'repeat': {'template': 'this is var%', 'for_each': {'var%': {'get_param': 'param'}}}}
    snippet_resolved = ['this is a', 'this is b', 'this is c']
    tmpl = template.Template(hot_tpl)
    stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
    self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl, stack))