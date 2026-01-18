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
def test_str_split_index_out_of_range(self):
    tmpl = template.Template(hot_liberty_tpl_empty)
    snippet = {'str_split': [',', 'bar,baz', '2']}
    exc = self.assertRaises(ValueError, self.resolve, snippet, tmpl)
    expected = 'Incorrect index to "str_split" should be between 0 and 1'
    self.assertEqual(expected, str(exc))