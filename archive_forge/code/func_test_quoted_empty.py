from .. import cmdline, tests
from .features import backslashdir_feature
def test_quoted_empty(self):
    self.assertAsTokens([(True, '')], '""')
    self.assertAsTokens([(False, "''")], "''")
    self.assertAsTokens([(True, '')], "''", single_quotes_allowed=True)
    self.assertAsTokens([(False, 'a'), (True, ''), (False, 'c')], 'a "" c')
    self.assertAsTokens([(False, 'a'), (True, ''), (False, 'c')], "a '' c", single_quotes_allowed=True)