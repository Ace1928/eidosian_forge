import os
import subprocess
from unittest import mock
import uuid
from oslo_policy import policy as common_policy
from keystone.common import policies
from keystone.common.rbac_enforcer import policy
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class PolicyJsonTestCase(unit.TestCase):

    def _get_default_policy_rules(self):
        """Return a dictionary of all in-code policies.

        All policies have a default value that is maintained in code.
        This method returns a dictionary containing all default policies.
        """
        rules = dict()
        for rule in policies.list_rules():
            rules[rule.name] = rule.check_str
        return rules

    def test_policies_loads(self):
        action = 'identity:list_projects'
        target = {'user_id': uuid.uuid4().hex, 'user.domain_id': uuid.uuid4().hex, 'group.domain_id': uuid.uuid4().hex, 'project.domain_id': uuid.uuid4().hex, 'project_id': uuid.uuid4().hex, 'domain_id': uuid.uuid4().hex}
        credentials = {'username': uuid.uuid4().hex, 'token': uuid.uuid4().hex, 'project_name': None, 'user_id': uuid.uuid4().hex, 'roles': [u'admin'], 'is_admin': True, 'is_admin_project': True, 'project_id': None, 'domain_id': uuid.uuid4().hex}
        result = policy._ENFORCER._enforcer.enforce(action, target, credentials)
        self.assertTrue(result)

    def test_all_targets_documented(self):
        policy_keys = self._get_default_policy_rules()
        policy_rule_keys = ['admin_or_owner', 'admin_or_token_subject', 'admin_required', 'owner', 'service_admin_or_token_subject', 'service_or_admin', 'service_role', 'token_subject']

        def read_doc_targets():
            doc_path = os.path.join(unit.ROOTDIR, 'doc', 'source', 'getting-started', 'policy_mapping.rst')
            with open(doc_path) as doc_file:
                for line in doc_file:
                    if line.startswith('Target'):
                        break
                for line in doc_file:
                    if line.startswith('==='):
                        break
                for line in doc_file:
                    line = line.rstrip()
                    if not line or line.startswith(' '):
                        continue
                    if line.startswith('=='):
                        break
                    target, dummy, dummy = line.partition(' ')
                    yield str(target)
        doc_targets = list(read_doc_targets())
        self.assertCountEqual(policy_keys, doc_targets + policy_rule_keys)