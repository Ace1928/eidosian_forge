from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.protorpclite import messages
from googlecloudsdk.core.resource import resource_expr_rewrite
from googlecloudsdk.core.util import times
import six
def _GuessOperandType(operand):
    """Returns the probable type for operand.

  This is a rewriter fallback, used if the resource proto message is not
  available.

  Args:
    operand: The operand string value to guess the type of.

  Returns:
    The probable type for the operand value.
  """
    try:
        int(operand)
    except ValueError:
        pass
    else:
        return int
    try:
        float(operand)
    except ValueError:
        pass
    else:
        return float
    if operand.lower() in ('true', 'false'):
        return bool
    if operand.replace('_', '').isupper():
        return messages.EnumField
    return six.text_type