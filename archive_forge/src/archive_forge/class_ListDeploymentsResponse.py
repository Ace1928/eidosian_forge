from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDeploymentsResponse(_messages.Message):
    """A ListDeploymentsResponse object.

  Fields:
    deployments: List of Deployments.
    nextPageToken: Token to be supplied to the next ListDeployments request
      via `page_token` to obtain the next set of results.
    unreachable: Locations that could not be reached.
  """
    deployments = _messages.MessageField('Deployment', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)