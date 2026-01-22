from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListHydratedDeploymentsResponse(_messages.Message):
    """Response object for `ListHydratedDeployments`.

  Fields:
    hydratedDeployments: The list of hydrated deployments.
    nextPageToken: A token that can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    hydratedDeployments = _messages.MessageField('HydratedDeployment', 1, repeated=True)
    nextPageToken = _messages.StringField(2)