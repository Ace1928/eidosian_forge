from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import contextlib
import itertools
import tokenize
from six import StringIO
from pasta.base import formatting as fmt
from pasta.base import fstring_utils
def whitespace(self, max_lines=None, comment=False):
    """Parses whitespace from the current _loc to the next non-whitespace.

    Arguments:
      max_lines: (optional int) Maximum number of lines to consider as part of
        the whitespace. Valid values are None, 0 and 1.
      comment: (boolean) If True, look for a trailing comment even when not in
        a parenthesized scope.

    Pre-condition:
      `_loc' represents the point before which everything has been parsed and
      after which nothing has been parsed.
    Post-condition:
      `_loc' is exactly at the character that was parsed to.
    """
    next_token = self.peek()
    if not comment and next_token and (next_token.type == TOKENS.COMMENT):
        return ''

    def predicate(token):
        return token.type in (TOKENS.INDENT, TOKENS.DEDENT) or (token.type == TOKENS.COMMENT and (comment or self._hints)) or (token.type == TOKENS.ERRORTOKEN and token.src == ' ') or (max_lines is None and token.type in (TOKENS.NL, TOKENS.NEWLINE))
    whitespace = list(self.takewhile(predicate, advance=False))
    next_token = self.peek()
    result = ''
    for tok in itertools.chain(whitespace, (next_token,) if next_token else ()):
        result += self._space_between(self._loc, tok.start)
        if tok != next_token:
            result += tok.src
            self._loc = tok.end
        else:
            self._loc = tok.start
    if (max_lines is None or max_lines > 0) and next_token and (next_token.type in (TOKENS.NL, TOKENS.NEWLINE)):
        result += self.next().src
    return result