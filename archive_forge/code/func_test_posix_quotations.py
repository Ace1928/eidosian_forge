from .. import cmdline, tests
from .features import backslashdir_feature
def test_posix_quotations(self):
    self.assertAsTokens([(True, 'foo bar')], "'foo bar'", single_quotes_allowed=True)
    self.assertAsTokens([(True, 'foo bar')], "'fo''o b''ar'", single_quotes_allowed=True)
    self.assertAsTokens([(True, 'foo bar')], '"fo""o b""ar"', single_quotes_allowed=True)
    self.assertAsTokens([(True, 'foo bar')], '"fo"\'o b\'"ar"', single_quotes_allowed=True)