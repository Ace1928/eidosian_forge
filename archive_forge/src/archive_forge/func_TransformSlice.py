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
def TransformSlice(r, op=':', undefined=''):
    """Returns a list slice specified by op.

  The op parameter consists of up to three colon-delimeted integers: start, end,
  and step. The parameter supports half-open ranges: start and end values can
  be omitted, representing the first and last positions of the resource
  respectively.

  The step value represents the increment between items in the resource included
  in the slice. A step of 2 results in a slice that contains every other item in
  the resource.

  Negative values for start and end indicate that the positons should start from
  the last position of the resource. A negative value for step indicates that
  the slice should contain items in reverse order.

  If op contains no colons, the slice consists of the single item at the
  specified position in the resource.

  Args:
    r: A JSON-serializable string or array.
    op: The slice operation.
    undefined: Returns this value if the slice cannot be created, or the
        resulting slice is empty.

  Returns:
    A new array containing the specified slice of the resource.

  Example:
    `[1,2,3].slice(1:)` returns `[2,3]`.

    `[1,2,3].slice(:2)` returns `[1,2]`.

    `[1,2,3].slice(-1:)` returns `[3]`.

    `[1,2,3].slice(: :-1)` returns `[3,2,1]`.

    `[1,2,3].slice(1)` returns `[2]`.
  """
    op = op.strip()
    if not op:
        return undefined
    try:
        ops = [int(sp) if sp else None for sp in (p.strip() for p in op.split(':'))]
    except (AttributeError, TypeError, ValueError):
        return undefined
    if len(ops) == 1:
        ops.append(ops[0] + 1 or None)
    try:
        return list(r[slice(*ops)]) or undefined
    except (TypeError, ValueError, KeyError):
        return undefined