from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
class TestRedirections(tests.TestCase):

    def _check(self, in_name, out_name, out_mode, remaining, args):
        self.assertEqual(script._scan_redirection_options(args), (in_name, out_name, out_mode, remaining))

    def test_no_redirection(self):
        self._check(None, None, None, [], [])
        self._check(None, None, None, ['foo', 'bar'], ['foo', 'bar'])

    def test_input_redirection(self):
        self._check('foo', None, None, [], ['<foo'])
        self._check('foo', None, None, ['bar'], ['bar', '<foo'])
        self._check('foo', None, None, ['bar'], ['bar', '<', 'foo'])
        self._check('foo', None, None, ['bar'], ['<foo', 'bar'])
        self._check('foo', None, None, ['bar', 'baz'], ['bar', '<foo', 'baz'])

    def test_output_redirection(self):
        self._check(None, 'foo', 'w+', [], ['>foo'])
        self._check(None, 'foo', 'w+', ['bar'], ['bar', '>foo'])
        self._check(None, 'foo', 'w+', ['bar'], ['bar', '>', 'foo'])
        self._check(None, 'foo', 'a+', [], ['>>foo'])
        self._check(None, 'foo', 'a+', ['bar'], ['bar', '>>foo'])
        self._check(None, 'foo', 'a+', ['bar'], ['bar', '>>', 'foo'])

    def test_redirection_syntax_errors(self):
        self._check('', None, None, [], ['<'])
        self._check(None, '', 'w+', [], ['>'])
        self._check(None, '', 'a+', [], ['>>'])
        self._check('>', '', 'a+', [], ['<', '>', '>>'])