from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
def TransformFloat(r, precision=6, spec=None, undefined=''):
    """Returns the string representation of a floating point number.

  One of these formats is used (1) ". _precision_ _spec_" if _spec_ is specified
  (2) ". _precision_" unless 1e-04 <= abs(number) < 1e+09 (3) ".1f" otherwise.

  Args:
    r: A JSON-serializable object.
    precision: The maximum number of digits before and after the decimal point.
    spec: The printf(3) floating point format "e", "f" or "g" spec character.
    undefined: Returns this value if the resource is not a float.

  Returns:
    The string representation of the floating point number r.
  """
    try:
        number = float(r)
    except (TypeError, ValueError):
        return undefined
    if spec is not None:
        fmt = '{{number:.{precision}{spec}}}'.format(precision=precision, spec=spec)
        return fmt.format(number=number)
    fmt = '{{number:.{precision}}}'.format(precision=precision)
    representation = fmt.format(number=number)
    exponent_index = representation.find('e+')
    if exponent_index >= 0:
        exponent = int(representation[exponent_index + 2:])
        if exponent < 9:
            return '{number:.1f}'.format(number=number)
    return representation