import os
import tempfile
from fire import __main__
from fire import testutils
class MainModuleTest(testutils.BaseTestCase):
    """Tests to verify the behavior of __main__ (python -m fire)."""

    def testNameSetting(self):
        with self.assertOutputMatches('gettempdir'):
            __main__.main(['__main__.py', 'tempfile'])

    def testArgPassing(self):
        expected = os.path.join('part1', 'part2', 'part3')
        with self.assertOutputMatches('%s\n' % expected):
            __main__.main(['__main__.py', 'os.path', 'join', 'part1', 'part2', 'part3'])
        with self.assertOutputMatches('%s\n' % expected):
            __main__.main(['__main__.py', 'os', 'path', '-', 'join', 'part1', 'part2', 'part3'])