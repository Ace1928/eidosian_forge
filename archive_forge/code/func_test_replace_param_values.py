import copy
import hashlib
import json
import fixtures
from stevedore import extension
from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import parameters as cfn_p
from heat.engine.cfn import template as cfn_t
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import template as hot_t
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_replace_param_values(self):
    env = environment.Environment({'foo': 'wibble'})
    tmpl = template.Template(parameter_template, env=env)
    stk = stack.Stack(self.ctx, 'test_stack', tmpl)
    snippet = {'Fn::Replace': [{'$var1': {'Ref': 'foo'}, '%var2%': {'Ref': 'blarg'}}, '$var1 is %var2%']}
    self.assertEqual('wibble is quux', self.resolve(snippet, tmpl, stk))