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
def test_str_replace_strict_no_missing_param(self):
    """Test str_replace_strict function no missing params, no problem."""
    snippet = {'str_replace_strict': {'template': 'Template var1 var1 s var2 t varvarvar3', 'params': {'var1': 'foo', 'var2': 'bar', 'var3': 'zed', 'var': 'tricky '}}}
    snippet_resolved = 'Template foo foo s bar t tricky tricky zed'
    tmpl = template.Template(hot_ocata_tpl_empty)
    self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl))