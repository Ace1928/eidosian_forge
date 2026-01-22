from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
import unicodedata
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
import six
class _ExprEQ(_ExprWordMatchBase):
    """EQ word match node."""

    def __init__(self, backend, key, operand, transform, op=None):
        super(_ExprEQ, self).__init__(backend, key, operand, transform, op=op or '=', warned_attribute='_deprecated_eq_warned')

    def _AddPattern(self, pattern):
        """Adds an EQ match pattern to self._patterns.

    A pattern is a word.

    This method re-implements both the original and the OnePlatform = using REs.
    It was tested against the original tests with no failures.  This cleaned up
    the code (really!) and made it easier to reason about the two
    implementations.

    Args:
      pattern: A string containing a word to match.
    """
        normalized_pattern = NormalizeForSearch(pattern)
        word = re.escape(normalized_pattern)
        standard_pattern = '\\b' + word + '\\b'
        deprecated_pattern = '^' + word + '$'
        reflags = re.IGNORECASE | re.MULTILINE | re.UNICODE
        standard_regex = _ReCompile(standard_pattern, reflags)
        deprecated_regex = _ReCompile(deprecated_pattern, reflags)
        self._patterns.append((pattern, standard_regex, deprecated_regex))