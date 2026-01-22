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
def TransformResolution(r, undefined='', transpose=False):
    """Formats a human readable XY resolution.

  Args:
    r: object, A JSON-serializable object containing an x/y resolution.
    undefined: Returns this value if a recognizable resolution was not found.
    transpose: Returns the y/x resolution if true.

  Returns:
    The human readable x/y resolution for r if it contains members that
      specify width/height, col/row, col/line, or x/y resolution. Returns
      undefined if no resolution found.
  """
    names = (('width', 'height'), ('screenx', 'screeny'), ('col', 'row'), ('col', 'line'), ('x', 'y'))
    mem = {}
    for m in r if isinstance(r, dict) else dir(r):
        if not m.startswith('__') and (not m.endswith('__')):
            mem[m.lower()] = m

    def _Dimension(d):
        """Gets the resolution dimension for d.

    Args:
      d: The dimension name substring to get.

    Returns:
      The resolution dimension matching d or None.
    """
        for m in mem:
            if d in m:
                return resource_property.Get(r, [mem[d]], None)
        return None
    for name_x, name_y in names:
        x = _Dimension(name_x)
        if x is None:
            continue
        y = _Dimension(name_y)
        if y is None:
            continue
        if GetBooleanArgValue(transpose):
            return '{y} x {x}'.format(x=x, y=y)
        return '{x} x {y}'.format(x=x, y=y)
    return undefined