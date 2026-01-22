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
class TestTemplateConditionParser(common.HeatTestCase):

    def setUp(self):
        super(TestTemplateConditionParser, self).setUp()
        self.ctx = utils.dummy_context()
        t = {'heat_template_version': '2016-10-14', 'parameters': {'env_type': {'type': 'string', 'default': 'test'}}, 'conditions': {'prod_env': {'equals': [{'get_param': 'env_type'}, 'prod']}}, 'resources': {'r1': {'type': 'GenericResourceType', 'condition': 'prod_env'}}, 'outputs': {'foo': {'condition': 'prod_env', 'value': 'show me'}}}
        self.tmpl = template.Template(t)

    def test_conditions_with_non_supported_functions(self):
        t = {'heat_template_version': '2016-10-14', 'parameters': {'env_type': {'type': 'string', 'default': 'test'}}, 'conditions': {'prod_env': {'equals': [{'get_param': 'env_type'}, {'get_attr': [None, 'att']}]}}}
        tmpl = template.Template(t)
        stk = stack.Stack(self.ctx, 'test_condition_with_get_attr_func', tmpl)
        ex = self.assertRaises(exception.StackValidationFailed, tmpl.conditions, stk)
        self.assertIn('"get_attr" is invalid', str(ex))
        self.assertIn('conditions.prod_env.equals[1].get_attr', str(ex))
        tmpl.t['conditions']['prod_env'] = {'get_resource': 'R1'}
        stk = stack.Stack(self.ctx, 'test_condition_with_get_attr_func', tmpl)
        ex = self.assertRaises(exception.StackValidationFailed, tmpl.conditions, stk)
        self.assertIn('"get_resource" is invalid', str(ex))
        tmpl.t['conditions']['prod_env'] = {'get_attr': [None, 'att']}
        stk = stack.Stack(self.ctx, 'test_condition_with_get_attr_func', tmpl)
        ex = self.assertRaises(exception.StackValidationFailed, tmpl.conditions, stk)
        self.assertIn('"get_attr" is invalid', str(ex))

    def test_condition_resolved_not_boolean(self):
        t = {'heat_template_version': '2016-10-14', 'parameters': {'env_type': {'type': 'string', 'default': 'test'}}, 'conditions': {'prod_env': {'get_param': 'env_type'}}}
        tmpl = template.Template(t)
        stk = stack.Stack(self.ctx, 'test_condition_not_boolean', tmpl)
        conditions = tmpl.conditions(stk)
        ex = self.assertRaises(exception.StackValidationFailed, conditions.is_enabled, 'prod_env')
        self.assertIn('The definition of condition "prod_env" is invalid', str(ex))

    def test_condition_reference_condition(self):
        t = {'heat_template_version': '2016-10-14', 'parameters': {'env_type': {'type': 'string', 'default': 'test'}}, 'conditions': {'prod_env': {'equals': [{'get_param': 'env_type'}, 'prod']}, 'test_env': {'not': 'prod_env'}, 'prod_or_test_env': {'or': ['prod_env', 'test_env']}, 'prod_and_test_env': {'and': ['prod_env', 'test_env']}}}
        tmpl = template.Template(t)
        stk = stack.Stack(self.ctx, 'test_condition_reference', tmpl)
        conditions = tmpl.conditions(stk)
        self.assertFalse(conditions.is_enabled('prod_env'))
        self.assertTrue(conditions.is_enabled('test_env'))
        self.assertTrue(conditions.is_enabled('prod_or_test_env'))
        self.assertFalse(conditions.is_enabled('prod_and_test_env'))

    def test_get_res_condition_invalid(self):
        tmpl = copy.deepcopy(self.tmpl)
        stk = stack.Stack(self.ctx, 'test_res_invalid_condition', tmpl)
        conds = tmpl.conditions(stk)
        ex = self.assertRaises(ValueError, conds.is_enabled, 'invalid_cd')
        self.assertIn('Invalid condition "invalid_cd"', str(ex))
        ex = self.assertRaises(ValueError, conds.is_enabled, 111)
        self.assertIn('Invalid condition "111"', str(ex))

    def test_res_condition_using_boolean(self):
        tmpl = copy.deepcopy(self.tmpl)
        stk = stack.Stack(self.ctx, 'test_res_cd_boolean', tmpl)
        conds = tmpl.conditions(stk)
        self.assertTrue(conds.is_enabled(True))
        self.assertFalse(conds.is_enabled(False))

    def test_parse_output_condition_invalid(self):
        stk = stack.Stack(self.ctx, 'test_output_invalid_condition', self.tmpl)
        self.tmpl.t['outputs']['foo']['condition'] = 'invalid_cd'
        ex = self.assertRaises(exception.StackValidationFailed, lambda: stk.outputs)
        self.assertIn('Invalid condition "invalid_cd"', str(ex))
        self.assertIn('outputs.foo.condition', str(ex))
        self.tmpl.t['outputs']['foo']['condition'] = 222
        ex = self.assertRaises(exception.StackValidationFailed, lambda: stk.outputs)
        self.assertIn('Invalid condition "222"', str(ex))
        self.assertIn('outputs.foo.condition', str(ex))

    def test_conditions_circular_ref(self):
        t = {'heat_template_version': '2016-10-14', 'parameters': {'env_type': {'type': 'string', 'default': 'test'}}, 'conditions': {'first_cond': {'not': 'second_cond'}, 'second_cond': {'not': 'third_cond'}, 'third_cond': {'not': 'first_cond'}}}
        tmpl = template.Template(t)
        stk = stack.Stack(self.ctx, 'test_condition_circular_ref', tmpl)
        conds = tmpl.conditions(stk)
        ex = self.assertRaises(exception.StackValidationFailed, conds.is_enabled, 'first_cond')
        self.assertIn('Circular definition for condition "first_cond"', str(ex))

    def test_parse_output_condition_boolean(self):
        t = copy.deepcopy(self.tmpl.t)
        t['outputs']['foo']['condition'] = True
        stk = stack.Stack(self.ctx, 'test_output_cd_boolean', template.Template(t))
        self.assertEqual('show me', stk.outputs['foo'].get_value())
        t = copy.deepcopy(self.tmpl.t)
        t['outputs']['foo']['condition'] = False
        stk = stack.Stack(self.ctx, 'test_output_cd_boolean', template.Template(t))
        self.assertIsNone(stk.outputs['foo'].get_value())

    def test_parse_output_condition_function(self):
        t = copy.deepcopy(self.tmpl.t)
        t['outputs']['foo']['condition'] = {'not': 'prod_env'}
        stk = stack.Stack(self.ctx, 'test_output_cd_function', template.Template(t))
        self.assertEqual('show me', stk.outputs['foo'].get_value())