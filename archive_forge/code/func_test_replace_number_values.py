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
def test_replace_number_values(self):
    tmpl = template.Template(empty_template)
    snippet = {'Fn::Replace': [{'$var1': 1, '%var2%': 2}, '$var1 is not %var2%']}
    self.assertEqual('1 is not 2', self.resolve(snippet, tmpl))
    snippet = {'Fn::Replace': [{'$var1': 1.3, '%var2%': 2.5}, '$var1 is not %var2%']}
    self.assertEqual('1.3 is not 2.5', self.resolve(snippet, tmpl))