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
def test_str_replace_single_pass(self):
    """Test that str_replace function does not do double substitution."""
    snippet = {'str_replace': {'template': '1234567890', 'params': {'1': 'a', '4': 'd', '8': 'h', '9': 'i', '123': '1', '456': '4', '890': '8', '90': '9'}}}
    tmpl = template.Template(hot_tpl_empty)
    self.assertEqual('1478', self.resolve(snippet, tmpl))