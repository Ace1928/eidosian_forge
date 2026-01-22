from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
def IsValidLabelValue(value):
    """Implements the PCRE r'[\\p{Ll}\\p{Lo}\\p{N}_-]{0,63}'.

  Only hyphens (-), underscores (_), lowercase characters, and numbers are
  allowed. International characters are allowed.

  Args:
    value: The label value, a string.

  Returns:
    True is the value is valid; False if not.
  """
    if value is None or len(value) > 63:
        return False
    return all((_IsValueOrSubsequent(c) for c in value))