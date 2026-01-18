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
def test_str_replace_missing_param(self):
    """Test str_replace function missing param is OK."""
    snippet = {'str_replace': {'template': 'Template var1 string var2', 'params': {'var1': 'foo', 'var2': 'bar', 'var3': 'zed'}}}
    snippet_resolved = 'Template foo string bar'
    for hot_tpl in (hot_tpl_empty, hot_ocata_tpl_empty):
        tmpl = template.Template(hot_tpl)
        self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))