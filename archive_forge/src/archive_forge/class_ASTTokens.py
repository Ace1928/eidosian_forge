import abc
import ast
import bisect
import sys
import token
from ast import Module
from typing import Iterable, Iterator, List, Optional, Tuple, Any, cast, TYPE_CHECKING
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from .line_numbers import LineNumbers
from .util import (
class ASTTokens(ASTTextBase, object):
    """
  ASTTokens maintains the text of Python code in several forms: as a string, as line numbers, and
  as tokens, and is used to mark and access token and position information.

  ``source_text`` must be a unicode or UTF8-encoded string. If you pass in UTF8 bytes, remember
  that all offsets you'll get are to the unicode text, which is available as the ``.text``
  property.

  If ``parse`` is set, the ``source_text`` will be parsed with ``ast.parse()``, and the resulting
  tree marked with token info and made available as the ``.tree`` property.

  If ``tree`` is given, it will be marked and made available as the ``.tree`` property. In
  addition to the trees produced by the ``ast`` module, ASTTokens will also mark trees produced
  using ``astroid`` library <https://www.astroid.org>.

  If only ``source_text`` is given, you may use ``.mark_tokens(tree)`` to mark the nodes of an AST
  tree created separately.
  """

    def __init__(self, source_text, parse=False, tree=None, filename='<unknown>', tokens=None):
        super(ASTTokens, self).__init__(source_text, filename)
        self._tree = ast.parse(source_text, filename) if parse else tree
        if tokens is None:
            tokens = generate_tokens(self._text)
        self._tokens = list(self._translate_tokens(tokens))
        self._token_offsets = [tok.startpos for tok in self._tokens]
        if self._tree:
            self.mark_tokens(self._tree)

    def mark_tokens(self, root_node):
        """
    Given the root of the AST or Astroid tree produced from source_text, visits all nodes marking
    them with token and position information by adding ``.first_token`` and
    ``.last_token``attributes. This is done automatically in the constructor when ``parse`` or
    ``tree`` arguments are set, but may be used manually with a separate AST or Astroid tree.
    """
        from .mark_tokens import MarkTokens
        MarkTokens(self).visit_tree(root_node)

    def _translate_tokens(self, original_tokens):
        """
    Translates the given standard library tokens into our own representation.
    """
        for index, tok in enumerate(patched_generate_tokens(original_tokens)):
            tok_type, tok_str, start, end, line = tok
            yield Token(tok_type, tok_str, start, end, line, index, self._line_numbers.line_to_offset(start[0], start[1]), self._line_numbers.line_to_offset(end[0], end[1]))

    @property
    def text(self):
        """The source code passed into the constructor."""
        return self._text

    @property
    def tokens(self):
        """The list of tokens corresponding to the source code from the constructor."""
        return self._tokens

    @property
    def tree(self):
        """The root of the AST tree passed into the constructor or parsed from the source code."""
        return self._tree

    @property
    def filename(self):
        """The filename that was parsed"""
        return self._filename

    def get_token_from_offset(self, offset):
        """
    Returns the token containing the given character offset (0-based position in source text),
    or the preceeding token if the position is between tokens.
    """
        return self._tokens[bisect.bisect(self._token_offsets, offset) - 1]

    def get_token(self, lineno, col_offset):
        """
    Returns the token containing the given (lineno, col_offset) position, or the preceeding token
    if the position is between tokens.
    """
        return self.get_token_from_offset(self._line_numbers.line_to_offset(lineno, col_offset))

    def get_token_from_utf8(self, lineno, col_offset):
        """
    Same as get_token(), but interprets col_offset as a UTF8 offset, which is what `ast` uses.
    """
        return self.get_token(lineno, self._line_numbers.from_utf8_col(lineno, col_offset))

    def next_token(self, tok, include_extra=False):
        """
    Returns the next token after the given one. If include_extra is True, includes non-coding
    tokens from the tokenize module, such as NL and COMMENT.
    """
        i = tok.index + 1
        if not include_extra:
            while is_non_coding_token(self._tokens[i].type):
                i += 1
        return self._tokens[i]

    def prev_token(self, tok, include_extra=False):
        """
    Returns the previous token before the given one. If include_extra is True, includes non-coding
    tokens from the tokenize module, such as NL and COMMENT.
    """
        i = tok.index - 1
        if not include_extra:
            while is_non_coding_token(self._tokens[i].type):
                i -= 1
        return self._tokens[i]

    def find_token(self, start_token, tok_type, tok_str=None, reverse=False):
        """
    Looks for the first token, starting at start_token, that matches tok_type and, if given, the
    token string. Searches backwards if reverse is True. Returns ENDMARKER token if not found (you
    can check it with `token.ISEOF(t.type)`).
    """
        t = start_token
        advance = self.prev_token if reverse else self.next_token
        while not match_token(t, tok_type, tok_str) and (not token.ISEOF(t.type)):
            t = advance(t, include_extra=True)
        return t

    def token_range(self, first_token, last_token, include_extra=False):
        """
    Yields all tokens in order from first_token through and including last_token. If
    include_extra is True, includes non-coding tokens such as tokenize.NL and .COMMENT.
    """
        for i in xrange(first_token.index, last_token.index + 1):
            if include_extra or not is_non_coding_token(self._tokens[i].type):
                yield self._tokens[i]

    def get_tokens(self, node, include_extra=False):
        """
    Yields all tokens making up the given node. If include_extra is True, includes non-coding
    tokens such as tokenize.NL and .COMMENT.
    """
        return self.token_range(node.first_token, node.last_token, include_extra=include_extra)

    def get_text_positions(self, node, padded):
        """
    Returns two ``(lineno, col_offset)`` tuples for the start and end of the given node.
    If the positions can't be determined, or the nodes don't correspond to any particular text,
    returns ``(1, 0)`` for both.

    ``padded`` corresponds to the ``padded`` argument to ``ast.get_source_segment()``.
    This means that if ``padded`` is True, the start position will be adjusted to include
    leading whitespace if ``node`` is a multiline statement.
    """
        if not hasattr(node, 'first_token'):
            return ((1, 0), (1, 0))
        start = node.first_token.start
        end = node.last_token.end
        if padded and any((match_token(t, token.NEWLINE) for t in self.get_tokens(node))):
            start = (start[0], 0)
        return (start, end)