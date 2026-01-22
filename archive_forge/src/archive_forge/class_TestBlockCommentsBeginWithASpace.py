import textwrap
import pycodestyle
from keystone.tests.hacking import checks
from keystone.tests import unit
from keystone.tests.unit.ksfixtures import hacking as hacking_fixtures
class TestBlockCommentsBeginWithASpace(BaseStyleCheck):

    def get_checker(self):
        return checks.block_comments_begin_with_a_space

    def test(self):
        code = self.code_ex.comments_begin_with_space['code']
        errors = self.code_ex.comments_begin_with_space['expected_errors']
        self.assert_has_errors(code, expected_errors=errors)