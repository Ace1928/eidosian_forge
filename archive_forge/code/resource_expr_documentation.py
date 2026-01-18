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
Adds an EQ match pattern to self._patterns.

    A pattern is a word.

    This method re-implements both the original and the OnePlatform = using REs.
    It was tested against the original tests with no failures.  This cleaned up
    the code (really!) and made it easier to reason about the two
    implementations.

    Args:
      pattern: A string containing a word to match.
    