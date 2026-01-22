from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsOperationsCancelRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsOperationsCancelRequest object.

  Fields:
    googleLongrunningCancelOperationRequest: A
      GoogleLongrunningCancelOperationRequest resource to be passed as the
      request body.
    name: The name of the operation resource to be cancelled.
  """
    googleLongrunningCancelOperationRequest = _messages.MessageField('GoogleLongrunningCancelOperationRequest', 1)
    name = _messages.StringField(2, required=True)