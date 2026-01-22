from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1SpotConfig(_messages.Message):
    """SpotConfig has configuration info for Spot node.

  Fields:
    instanceTypes: Required. A list of instance types for creating spot node
      pool.
  """
    instanceTypes = _messages.StringField(1, repeated=True)