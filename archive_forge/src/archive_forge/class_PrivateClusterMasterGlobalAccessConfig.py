from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivateClusterMasterGlobalAccessConfig(_messages.Message):
    """Configuration for controlling master global access settings.

  Fields:
    enabled: Whenever master is accessible globally or not.
  """
    enabled = _messages.BooleanField(1)