from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPostureDeploymentsResponse(_messages.Message):
    """Message for response to listing PostureDeployments.

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    postureDeployments: The list of PostureDeployment.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    postureDeployments = _messages.MessageField('PostureDeployment', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)