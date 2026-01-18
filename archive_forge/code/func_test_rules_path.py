import sys
from breezy import rules, tests
def test_rules_path(self):
    self.assertEqual(rules.rules_path(), self.brz_home + '/rules')