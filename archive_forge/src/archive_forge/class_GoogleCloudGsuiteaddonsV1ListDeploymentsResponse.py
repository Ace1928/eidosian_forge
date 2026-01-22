from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGsuiteaddonsV1ListDeploymentsResponse(_messages.Message):
    """Response message to list deployments.

  Fields:
    deployments: The list of deployments for the given project.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    deployments = _messages.MessageField('GoogleCloudGsuiteaddonsV1Deployment', 1, repeated=True)
    nextPageToken = _messages.StringField(2)