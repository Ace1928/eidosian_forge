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
def test_str_replace_ref_get_param(self):
    """Test str_replace referencing parameters."""
    hot_tpl = template_format.parse('\n        heat_template_version: 2015-04-30\n        parameters:\n          p_template:\n            type: string\n            default: foo-replaceme\n          p_params:\n            type: json\n            default:\n              replaceme: success\n        resources:\n          rsrc:\n            type: ResWithStringPropAndAttr\n            properties:\n              a_string:\n                str_replace:\n                  template: {get_param: p_template}\n                  params: {get_param: p_params}\n        outputs:\n          replaced:\n            value: {get_attr: [rsrc, string]}\n        ')
    tmpl = template.Template(hot_tpl)
    self.stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
    self.stack.store()
    self.stack.create()
    self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), self.stack.state)
    self.stack._update_all_resource_data(False, True)
    self.assertEqual('foo-success', self.stack.outputs['replaced'].get_value())