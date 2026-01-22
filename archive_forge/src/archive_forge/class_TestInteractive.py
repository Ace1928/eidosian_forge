import cmd2
from cliff.interactive import InteractiveApp
from cliff.tests import base
class TestInteractive(base.TestBase):

    def make_interactive_app(self, errexit, *command_names):
        fake_command_manager = [(x, None) for x in command_names]
        return InteractiveApp(FakeApp, fake_command_manager, stdin=None, stdout=None, errexit=errexit)

    def _test_completenames(self, expecteds, prefix):
        app = self.make_interactive_app(False, 'hips', 'hippo', 'nonmatching')
        self.assertEqual(set(app.completenames(prefix, '', 0, 1)), set(expecteds))

    def test_cmd2_completenames(self):
        self._test_completenames(['help'], 'he')

    def test_cliff_completenames(self):
        self._test_completenames(['hips', 'hippo'], 'hip')

    def test_no_completenames(self):
        self._test_completenames([], 'taz')

    def test_both_completenames(self):
        if hasattr(cmd2.Cmd, 'do_hi'):
            self._test_completenames(['hi', 'history', 'hips', 'hippo'], 'hi')
        else:
            self._test_completenames(['history', 'hips', 'hippo'], 'hi')

    def _test_completedefault(self, expecteds, line, begidx):
        command_names = set(['show file', 'show folder', 'show  long', 'list all'])
        app = self.make_interactive_app(False, *command_names)
        observeds = app.completedefault(None, line, begidx, None)
        self.assertEqual(set(expecteds), set(observeds))
        self.assertTrue(set([line[:begidx] + x for x in observeds]) <= command_names)

    def test_empty_text_completedefault(self):
        self._test_completedefault(['file', 'folder', ' long'], 'show ', 5)

    def test_nonempty_text_completedefault2(self):
        self._test_completedefault(['file', 'folder'], 'show f', 5)

    def test_long_completedefault(self):
        self._test_completedefault(['long'], 'show  ', 6)

    def test_no_completedefault(self):
        self._test_completedefault([], 'taz ', 4)

    def test_no_errexit(self):
        command_names = set(['show file', 'show folder', 'list all'])
        app = self.make_interactive_app(False, *command_names)
        self.assertFalse(app.errexit)

    def test_errexit(self):
        command_names = set(['show file', 'show folder', 'list all'])
        app = self.make_interactive_app(True, *command_names)
        self.assertTrue(app.errexit)