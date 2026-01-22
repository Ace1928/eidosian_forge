import yaml
from heat_integrationtests.functional import functional_base
class DefaultParametersTest(functional_base.FunctionalTestsBase):
    template = '\nheat_template_version: 2013-05-23\nparameters:\n  length:\n    type: string\n    default: 40\nresources:\n  random1:\n    type: nested_random.yaml\n  random2:\n    type: OS::Heat::RandomString\n    properties:\n      length: {get_param: length}\noutputs:\n  random1:\n    value: {get_attr: [random1, random1_value]}\n  random2:\n    value: {get_resource: random2}\n'
    nested_template = '\nheat_template_version: 2013-05-23\nparameters:\n  length:\n    type: string\n    default: 50\nresources:\n  random1:\n    type: OS::Heat::RandomString\n    properties:\n      length: {get_param: length}\noutputs:\n  random1_value:\n    value: {get_resource: random1}\n'
    scenarios = [('none', dict(param=None, default=None, temp_def=True, expect1=50, expect2=40)), ('default', dict(param=None, default=12, temp_def=True, expect1=12, expect2=12)), ('both', dict(param=15, default=12, temp_def=True, expect1=12, expect2=15)), ('no_temp_default', dict(param=None, default=12, temp_def=False, expect1=12, expect2=12))]

    def test_defaults(self):
        env = {'parameters': {}, 'parameter_defaults': {}}
        if self.param:
            env['parameters'] = {'length': self.param}
        if self.default:
            env['parameter_defaults'] = {'length': self.default}
        if not self.temp_def:
            ntempl = yaml.safe_load(self.nested_template)
            del ntempl['parameters']['length']['default']
            nested_template = yaml.safe_dump(ntempl)
        else:
            nested_template = self.nested_template
        stack_identifier = self.stack_create(template=self.template, files={'nested_random.yaml': nested_template}, environment=env)
        stack = self.client.stacks.get(stack_identifier)
        for out in stack.outputs:
            if out['output_key'] == 'random1':
                self.assertEqual(self.expect1, len(out['output_value']))
            if out['output_key'] == 'random2':
                self.assertEqual(self.expect2, len(out['output_value']))