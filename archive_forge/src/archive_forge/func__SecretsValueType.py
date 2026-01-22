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
def _SecretsValueType(value):
    """Validates secrets value configuration.

  The restrictions for gcloud are strict when compared to GCF to accommodate
  future changes without making it confusing for the user.

  Args:
    value: Secrets value configuration.

  Returns:
    Secrets value configuration as a string.

  Raises:
    ArgumentTypeError: Secrets value configuration is not valid.
  """
    if '=' in value:
        raise ArgumentTypeError("Secrets value configuration cannot contain '=' [{}]".format(value))
    return _CanonicalizeValue(value)