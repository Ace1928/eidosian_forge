import textwrap
import pycodestyle
from keystone.tests.hacking import checks
from keystone.tests import unit
from keystone.tests.unit.ksfixtures import hacking as hacking_fixtures
class TestTranslationChecks(BaseStyleCheck):

    def get_checker(self):
        return checks.CheckForTranslationIssues

    def get_fixture(self):
        return hacking_fixtures.HackingTranslations()

    def assert_has_errors(self, code, expected_errors=None):
        actual_errors = (e[:3] for e in self.run_check(code))
        import_lines = len(self.code_ex.shared_imports.split('\n')) - 1
        actual_errors = [(e[0] - import_lines, e[1], e[2]) for e in actual_errors]
        self.assertEqual(expected_errors or [], actual_errors)

    def test_for_translations(self):
        for example in self.code_ex.examples:
            code = self.code_ex.shared_imports + example['code']
            errors = example['expected_errors']
            self.assert_has_errors(code, expected_errors=errors)