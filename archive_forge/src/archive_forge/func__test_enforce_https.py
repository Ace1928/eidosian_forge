import json
from oslo_policy import fixture
from oslo_policy import policy as oslo_policy
from oslo_policy.tests import base as test_base
def _test_enforce_https(self, return_value):
    self.useFixture(fixture.HttpsCheckFixture(return_value=return_value))
    action = self.getUniqueString()
    rules_json = {action: 'https:' + self.getUniqueString()}
    rules = oslo_policy.Rules.load(json.dumps(rules_json))
    self.enforcer.set_rules(rules)
    return self.enforcer.enforce(rule=action, target={}, creds={})