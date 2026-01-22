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
def _TransformIsEnabled(self, transform):
    """Returns True if transform is enabled.

    Args:
      transform: The resource_projection_parser._Transform object.

    Returns:
      True if the transform is enabled, False if not.
    """
    if self._transforms_enabled is not None:
        return self._transforms_enabled
    return transform.active in (None, self._projection.active)