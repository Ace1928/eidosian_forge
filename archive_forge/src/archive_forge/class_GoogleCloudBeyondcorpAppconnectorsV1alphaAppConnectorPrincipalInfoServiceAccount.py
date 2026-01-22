from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpAppconnectorsV1alphaAppConnectorPrincipalInfoServiceAccount(_messages.Message):
    """ServiceAccount represents a GCP service account.

  Fields:
    email: Email address of the service account.
  """
    email = _messages.StringField(1)