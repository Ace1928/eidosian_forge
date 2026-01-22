from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureClusterError(_messages.Message):
    """AzureClusterError describes errors found on Azure clusters.

  Fields:
    message: Human-friendly description of the error.
  """
    message = _messages.StringField(1)