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
def test_repeat_bad_args(self):
    """Tests reporting error by repeat function.

        Test that the repeat function reports a proper error when missing or
        invalid arguments.
        """
    tmpl = template.Template(hot_kilo_tpl_empty)
    snippet = {'repeat': {'template': 'this is %var%'}}
    self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)
    snippet = {'repeat': {'template': 'this is %var%', 'foreach': {'%var%': ['a', 'b', 'c']}}}
    self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)
    snippet = {'repeat': {'templte': 'this is %var%', 'for_each': {'%var%': ['a', 'b', 'c']}}}
    self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)