import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
class TemplateResourceUpdateTest(functional_base.FunctionalTestsBase):
    """Prove that we can do template resource updates."""
    main_template = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  the_nested:\n    Type: the.yaml\n    Properties:\n      one: my_name\n      two: your_name\nOutputs:\n  identifier:\n    Value: {Ref: the_nested}\n  value:\n    Value: {'Fn::GetAtt': [the_nested, the_str]}\n"
    main_template_change_prop = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  the_nested:\n    Type: the.yaml\n    Properties:\n      one: updated_name\n      two: your_name\n\nOutputs:\n  identifier:\n    Value: {Ref: the_nested}\n  value:\n    Value: {'Fn::GetAtt': [the_nested, the_str]}\n"
    main_template_add_prop = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  the_nested:\n    Type: the.yaml\n    Properties:\n      one: my_name\n      two: your_name\n      three: third_name\n\nOutputs:\n  identifier:\n    Value: {Ref: the_nested}\n  value:\n    Value: {'Fn::GetAtt': [the_nested, the_str]}\n"
    main_template_remove_prop = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  the_nested:\n    Type: the.yaml\n    Properties:\n      one: my_name\n\nOutputs:\n  identifier:\n    Value: {Ref: the_nested}\n  value:\n    Value: {'Fn::GetAtt': [the_nested, the_str]}\n"
    initial_tmpl = "\nHeatTemplateFormatVersion: '2012-12-12'\nParameters:\n  one:\n    Default: foo\n    Type: String\n  two:\n    Default: bar\n    Type: String\n\nResources:\n  NestedResource:\n    Type: OS::Heat::RandomString\n    Properties:\n      salt: {Ref: one}\nOutputs:\n  the_str:\n    Value: {'Fn::GetAtt': [NestedResource, value]}\n"
    prop_change_tmpl = "\nHeatTemplateFormatVersion: '2012-12-12'\nParameters:\n  one:\n    Default: yikes\n    Type: String\n  two:\n    Default: foo\n    Type: String\nResources:\n  NestedResource:\n    Type: OS::Heat::RandomString\n    Properties:\n      salt: {Ref: two}\nOutputs:\n  the_str:\n    Value: {'Fn::GetAtt': [NestedResource, value]}\n"
    prop_add_tmpl = "\nHeatTemplateFormatVersion: '2012-12-12'\nParameters:\n  one:\n    Default: yikes\n    Type: String\n  two:\n    Default: foo\n    Type: String\n  three:\n    Default: bar\n    Type: String\n\nResources:\n  NestedResource:\n    Type: OS::Heat::RandomString\n    Properties:\n      salt: {Ref: three}\nOutputs:\n  the_str:\n    Value: {'Fn::GetAtt': [NestedResource, value]}\n"
    prop_remove_tmpl = "\nHeatTemplateFormatVersion: '2012-12-12'\nParameters:\n  one:\n    Default: yikes\n    Type: String\n\nResources:\n  NestedResource:\n    Type: OS::Heat::RandomString\n    Properties:\n      salt: {Ref: one}\nOutputs:\n  the_str:\n    Value: {'Fn::GetAtt': [NestedResource, value]}\n"
    attr_change_tmpl = "\nHeatTemplateFormatVersion: '2012-12-12'\nParameters:\n  one:\n    Default: foo\n    Type: String\n  two:\n    Default: bar\n    Type: String\n\nResources:\n  NestedResource:\n    Type: OS::Heat::RandomString\n    Properties:\n      salt: {Ref: one}\nOutputs:\n  the_str:\n    Value: {'Fn::GetAtt': [NestedResource, value]}\n  something_else:\n    Value: just_a_string\n"
    content_change_tmpl = "\nHeatTemplateFormatVersion: '2012-12-12'\nParameters:\n  one:\n    Default: foo\n    Type: String\n  two:\n    Default: bar\n    Type: String\n\nResources:\n  NestedResource:\n    Type: OS::Heat::RandomString\n    Properties:\n      salt: yum\nOutputs:\n  the_str:\n    Value: {'Fn::GetAtt': [NestedResource, value]}\n"
    EXPECTED = UPDATE, NOCHANGE = ('update', 'nochange')
    scenarios = [('no_changes', dict(template=main_template, provider=initial_tmpl, expect=NOCHANGE)), ('main_tmpl_change', dict(template=main_template_change_prop, provider=initial_tmpl, expect=UPDATE)), ('provider_change', dict(template=main_template, provider=content_change_tmpl, expect=UPDATE)), ('provider_props_change', dict(template=main_template, provider=prop_change_tmpl, expect=UPDATE)), ('provider_props_add', dict(template=main_template_add_prop, provider=prop_add_tmpl, expect=UPDATE)), ('provider_props_remove', dict(template=main_template_remove_prop, provider=prop_remove_tmpl, expect=NOCHANGE)), ('provider_attr_change', dict(template=main_template, provider=attr_change_tmpl, expect=NOCHANGE))]

    def test_template_resource_update_template_schema(self):
        stack_identifier = self.stack_create(template=self.main_template, files={'the.yaml': self.initial_tmpl})
        stack = self.client.stacks.get(stack_identifier)
        initial_id = self._stack_output(stack, 'identifier')
        initial_val = self._stack_output(stack, 'value')
        self.update_stack(stack_identifier, self.template, files={'the.yaml': self.provider})
        stack = self.client.stacks.get(stack_identifier)
        self.assertEqual(initial_id, self._stack_output(stack, 'identifier'))
        if self.expect == self.NOCHANGE:
            self.assertEqual(initial_val, self._stack_output(stack, 'value'))
        else:
            self.assertNotEqual(initial_val, self._stack_output(stack, 'value'))