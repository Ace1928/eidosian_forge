import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
class TemplateResourceAdoptTest(functional_base.FunctionalTestsBase):
    """Prove that we can do template resource adopt/abandon."""
    main_template = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  the_nested:\n    Type: the.yaml\n    Properties:\n      one: my_name\nOutputs:\n  identifier:\n    Value: {Ref: the_nested}\n  value:\n    Value: {'Fn::GetAtt': [the_nested, the_str]}\n"
    nested_templ = "\nHeatTemplateFormatVersion: '2012-12-12'\nParameters:\n  one:\n    Default: foo\n    Type: String\nResources:\n  RealRandom:\n    Type: OS::Heat::RandomString\n    Properties:\n      salt: {Ref: one}\nOutputs:\n  the_str:\n    Value: {'Fn::GetAtt': [RealRandom, value]}\n"

    def _yaml_to_json(self, yaml_templ):
        return yaml.safe_load(yaml_templ)

    def test_abandon(self):
        stack_identifier = self.stack_create(template=self.main_template, files={'the.yaml': self.nested_templ}, enable_cleanup=False)
        info = self.stack_abandon(stack_id=stack_identifier)
        self.assertEqual(self._yaml_to_json(self.main_template), info['template'])
        self.assertEqual(self._yaml_to_json(self.nested_templ), info['resources']['the_nested']['template'])

    def test_adopt(self):
        data = {'resources': {'the_nested': {'type': 'the.yaml', 'resources': {'RealRandom': {'type': 'OS::Heat::RandomString', 'resource_data': {'value': 'goopie'}, 'resource_id': 'froggy'}}}}, 'environment': {'parameters': {}}, 'template': yaml.safe_load(self.main_template)}
        stack_identifier = self.stack_adopt(adopt_data=json.dumps(data), files={'the.yaml': self.nested_templ})
        self.assert_resource_is_a_stack(stack_identifier, 'the_nested')
        stack = self.client.stacks.get(stack_identifier)
        self.assertEqual('goopie', self._stack_output(stack, 'value'))