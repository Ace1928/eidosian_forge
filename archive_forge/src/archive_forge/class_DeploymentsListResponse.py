from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentsListResponse(_messages.Message):
    """A response containing a partial list of deployments and a page token
  used to build the next request if the request has been truncated.

  Fields:
    deployments: Output only. The deployments contained in this response.
    nextPageToken: Output only. A token used to continue a truncated list
      request.
  """
    deployments = _messages.MessageField('Deployment', 1, repeated=True)
    nextPageToken = _messages.StringField(2)