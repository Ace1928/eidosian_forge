from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.arg_parsers import ArgumentTypeError
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import log
import six
def CanonicalizeKey(key):
    """Canonicalizes secrets configuration key.

  Args:
    key: Secrets configuration key.

  Returns:
    Canonicalized secrets configuration key.
  """
    if _SECRET_PATH_PATTERN.search(key):
        return _CanonicalizePath(key)
    return key