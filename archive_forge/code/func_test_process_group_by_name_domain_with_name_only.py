import copy
import uuid
from keystone.exception import ValidationError
from keystone.federation import utils
from keystone.tests import unit
def test_process_group_by_name_domain_with_name_only(self):
    domain = {'name': 'domain1'}
    group1 = {'name': 'group1', 'domain': domain}
    group_by_domain = {}
    result = self.rule_processor.process_group_by_name(group1, group_by_domain)
    self.assertEqual([group1], list(result))
    self.assertEqual([domain['name']], list(group_by_domain.keys()))