from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDeploymentRevisionsResponse(_messages.Message):
    """List of deployment revisions for a given deployment.

  Fields:
    deployments: The revisions of the deployment.
    nextPageToken: A token that can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    deployments = _messages.MessageField('Deployment', 1, repeated=True)
    nextPageToken = _messages.StringField(2)