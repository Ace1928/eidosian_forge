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
def SetRetainNoneValues(self, enable):
    """Sets the projection to retain-none-values mode.

    Args:
      enable: Enables projection to a retain-none-values if True.
    """
    self._retain_none_values = enable