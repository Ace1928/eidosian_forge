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
def _ProjectAttribute(self, obj, projection, flag):
    """Applies projection.attribute.transform in projection if any to obj.

    Args:
      obj: An object.
      projection: Projection _Tree node.
      flag: A bitmask of DEFAULT, INNER, PROJECT.

    Returns:
      The transformed obj if there was a transform, otherwise the original obj.
    """
    if flag < self._projection.PROJECT:
        return None
    if projection and projection.attribute and projection.attribute.transform and self._TransformIsEnabled(projection.attribute.transform):
        return projection.attribute.transform.Evaluate(obj)
    return self._Project(obj, projection, flag, leaf=True)