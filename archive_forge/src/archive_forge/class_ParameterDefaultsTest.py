from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
class ParameterDefaultsTest(ParametersBase):
    scenarios = [('type_list', dict(p_type='CommaDelimitedList', value='1,1,1', expected=[['4', '2'], ['7', '7'], ['1', '1', '1']], param_default='7,7', default='4,2')), ('type_number', dict(p_type='Number', value=111, expected=[42, 77, 111], param_default=77, default=42)), ('type_string', dict(p_type='String', value='111', expected=['42', '77', '111'], param_default='77', default='42')), ('type_json', dict(p_type='Json', value={'1': '11'}, expected=[{'4': '2'}, {'7': '7'}, {'1': '11'}], param_default={'7': '7'}, default={'4': '2'})), ('type_boolean1', dict(p_type='Boolean', value=True, expected=[False, False, True], param_default=False, default=False)), ('type_boolean2', dict(p_type='Boolean', value=False, expected=[False, True, False], param_default=True, default=False)), ('type_boolean3', dict(p_type='Boolean', value=False, expected=[True, False, False], param_default=False, default=True))]

    def test_use_expected_default(self):
        template = {'Parameters': {'a': {'Type': self.p_type, 'Default': self.default}}}
        params = self.new_parameters('test_params', template)
        self.assertEqual(self.expected[0], params['a'])
        params = self.new_parameters('test_params', template, param_defaults={'a': self.param_default})
        self.assertEqual(self.expected[1], params['a'])
        params = self.new_parameters('test_params', template, {'a': self.value}, param_defaults={'a': self.param_default})
        self.assertEqual(self.expected[2], params['a'])