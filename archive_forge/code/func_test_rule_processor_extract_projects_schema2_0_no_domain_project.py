import copy
import uuid
from keystone.exception import ValidationError
from keystone.federation import utils
from keystone.tests import unit
def test_rule_processor_extract_projects_schema2_0_no_domain_project(self):
    project = {'name': 'project1'}
    identity_values = {'projects': [project.copy()], 'domain': self.domain_mock}
    result = self.rule_processor_schema_2_0.extract_projects(identity_values)
    expected_project = project.copy()
    expected_project['domain'] = self.domain_mock
    self.assertEqual([expected_project], result)