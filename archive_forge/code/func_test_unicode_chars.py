from .. import cmdline, tests
from .features import backslashdir_feature
def test_unicode_chars(self):
    self.assertAsTokens([(False, 'fµî'), (False, 'ሴ㑖')], 'fµî ሴ㑖')