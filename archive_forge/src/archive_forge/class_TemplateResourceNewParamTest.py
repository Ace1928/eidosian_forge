import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
class TemplateResourceNewParamTest(functional_base.FunctionalTestsBase):
    main_template = '\nheat_template_version: 2013-05-23\nresources:\n  my_resource:\n    type: resource.yaml\n    properties:\n      value1: foo\n'
    nested_templ = '\nheat_template_version: 2013-05-23\nparameters:\n  value1:\n    type: string\nresources:\n  test:\n    type: OS::Heat::TestResource\n    properties:\n      value: {get_param: value1}\n'
    main_template_update = '\nheat_template_version: 2013-05-23\nresources:\n  my_resource:\n    type: resource.yaml\n    properties:\n      value1: foo\n      value2: foo\n'
    nested_templ_update_fail = '\nheat_template_version: 2013-05-23\nparameters:\n  value1:\n    type: string\n  value2:\n    type: string\nresources:\n  test:\n    type: OS::Heat::TestResource\n    properties:\n      fail: True\n      value:\n        str_replace:\n          template: VAL1-VAL2\n          params:\n            VAL1: {get_param: value1}\n            VAL2: {get_param: value2}\n'
    nested_templ_update = '\nheat_template_version: 2013-05-23\nparameters:\n  value1:\n    type: string\n  value2:\n    type: string\nresources:\n  test:\n    type: OS::Heat::TestResource\n    properties:\n      value:\n        str_replace:\n          template: VAL1-VAL2\n          params:\n            VAL1: {get_param: value1}\n            VAL2: {get_param: value2}\n'

    def test_update(self):
        stack_identifier = self.stack_create(template=self.main_template, files={'resource.yaml': self.nested_templ})
        self.update_stack(stack_identifier, self.main_template_update, files={'resource.yaml': self.nested_templ_update_fail}, expected_status='UPDATE_FAILED')
        self.update_stack(stack_identifier, self.main_template_update, files={'resource.yaml': self.nested_templ_update})