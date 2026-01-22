from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigConnectorConfig(_messages.Message):
    """Configuration options for the Config Connector add-on.

  Fields:
    enabled: Whether Cloud Connector is enabled for this cluster.
  """
    enabled = _messages.BooleanField(1)