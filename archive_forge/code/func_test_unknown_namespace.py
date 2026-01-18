import sys
from breezy import rules, tests
def test_unknown_namespace(self):
    self.assertRaises(rules.UnknownRules, rules._IniBasedRulesSearcher, ['[junk]', 'foo=bar'])