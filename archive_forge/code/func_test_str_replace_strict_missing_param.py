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
def test_str_replace_strict_missing_param(self):
    """Test str_replace_strict function missing param(s) raises error."""
    snippet = {'str_replace_strict': {'template': 'Template var1 string var2', 'params': {'var1': 'foo', 'var2': 'bar', 'var3': 'zed'}}}
    tmpl = template.Template(hot_ocata_tpl_empty)
    ex = self.assertRaises(ValueError, self.resolve, snippet, tmpl)
    self.assertEqual('The following params were not found in the template: var3', str(ex))
    snippet = {'str_replace_strict': {'template': 'Template var1 string var2', 'params': {'var1': 'foo', 'var2': 'bar', 'var0': 'zed'}}}
    ex = self.assertRaises(ValueError, self.resolve, snippet, tmpl)
    self.assertEqual('The following params were not found in the template: var0', str(ex))
    snippet = {'str_replace_vstrict': {'template': 'Template var1 string var2', 'params': {'var1': 'foo', 'var2': 'bar', 'var0': 'zed', 'var': 'z', 'longvarname': 'q'}}}
    tmpl = template.Template(hot_pike_tpl_empty)
    ex = self.assertRaises(ValueError, self.resolve, snippet, tmpl)
    self.assertEqual('The following params were not found in the template: longvarname,var0,var', str(ex))