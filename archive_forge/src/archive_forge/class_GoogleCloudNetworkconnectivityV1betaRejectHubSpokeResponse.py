from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworkconnectivityV1betaRejectHubSpokeResponse(_messages.Message):
    """The response for HubService.RejectHubSpoke.

  Fields:
    spoke: The spoke that was operated on.
  """
    spoke = _messages.MessageField('GoogleCloudNetworkconnectivityV1betaSpoke', 1)