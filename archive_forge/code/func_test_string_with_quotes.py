from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import lex
from cmakelang.lex import TokenType
def test_string_with_quotes(self):
    self.assert_tok_types('\n      "this is a \\"string"\n      ', [TokenType.NEWLINE, TokenType.WHITESPACE, TokenType.QUOTED_LITERAL, TokenType.NEWLINE, TokenType.WHITESPACE])
    self.assert_tok_types("\n      'this is a \\'string'\n      ", [TokenType.NEWLINE, TokenType.WHITESPACE, TokenType.QUOTED_LITERAL, TokenType.NEWLINE, TokenType.WHITESPACE])