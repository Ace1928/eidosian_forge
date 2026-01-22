from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import json
from apitools.base.protorpclite import messages as protorpc_message
from apitools.base.py import encoding as protorpc_encoding
from googlecloudsdk.core.resource import resource_projection_parser
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _ProjectClass(self, obj, projection, flag):
    """Converts class object to a dict.

    Private and callable attributes are omitted in the dict.

    Args:
      obj: The class object to convert.
      projection: Projection _Tree node.
      flag: A bitmask of DEFAULT, INNER, PROJECT.

    Returns:
      The dict representing the class object.
    """
    r = {}
    exclude = set()
    if isinstance(obj, datetime.datetime):
        r['datetime'] = six.text_type(obj)
        exclude.update(('max', 'min', 'resolution', 'tzinfo'))
    else:
        try:
            exclude.update([a for a in dir(obj.__class__) if a.isupper()])
        except AttributeError:
            pass
    for attr in dir(obj):
        if attr.startswith('_'):
            continue
        if attr in exclude:
            continue
        try:
            value = getattr(obj, attr)
        except:
            continue
        if hasattr(value, '__call__'):
            continue
        f = flag
        if attr in projection.tree:
            child_projection = projection.tree[attr]
            f |= child_projection.attribute.flag
            if f < self._projection.INNER:
                continue
            r[attr] = self._Project(value, child_projection, f)
        else:
            r[attr] = self._ProjectAttribute(value, self._projection.GetEmpty(), f)
    return r