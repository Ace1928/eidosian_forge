from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_redirection_syntax_errors(self):
    self._check('', None, None, [], ['<'])
    self._check(None, '', 'w+', [], ['>'])
    self._check(None, '', 'a+', [], ['>>'])
    self._check('>', '', 'a+', [], ['<', '>', '>>'])