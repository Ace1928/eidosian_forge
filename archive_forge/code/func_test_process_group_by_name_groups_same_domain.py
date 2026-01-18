import copy
import uuid
from keystone.exception import ValidationError
from keystone.federation import utils
from keystone.tests import unit
def test_process_group_by_name_groups_same_domain(self):
    group1 = {'name': 'group1', 'domain': self.domain_mock}
    group2 = {'name': 'group2', 'domain': self.domain_mock}
    group_by_domain = {self.domain_id_mock: [group1]}
    result = self.rule_processor.process_group_by_name(group2, group_by_domain)
    self.assertEqual([group1, group2], list(result))
    self.assertEqual([self.domain_id_mock], list(group_by_domain.keys()))