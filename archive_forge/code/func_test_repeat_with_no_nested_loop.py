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
def test_repeat_with_no_nested_loop(self):
    snippet = {'repeat': {'template': {'network': '%net%', 'port': '%port%', 'subnet': '%sub%'}, 'for_each': {'%net%': ['n1', 'n2', 'n3', 'n4'], '%port%': ['p1', 'p2', 'p3', 'p4'], '%sub%': ['s1', 's2', 's3', 's4']}, 'permutations': False}}
    tmpl = template.Template(hot_pike_tpl_empty)
    snippet_resolved = [{'network': 'n1', 'port': 'p1', 'subnet': 's1'}, {'network': 'n2', 'port': 'p2', 'subnet': 's2'}, {'network': 'n3', 'port': 'p3', 'subnet': 's3'}, {'network': 'n4', 'port': 'p4', 'subnet': 's4'}]
    result = self.resolve(snippet, tmpl)
    self.assertEqual(snippet_resolved, result)