from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListFeaturesResponse(_messages.Message):
    """Response message for the `GkeHub.ListFeatures` method.

  Fields:
    nextPageToken: A token to request the next page of resources from the
      `ListFeatures` method. The value of an empty string means that there are
      no more resources to return.
    resources: The list of matching Features
  """
    nextPageToken = _messages.StringField(1)
    resources = _messages.MessageField('Feature', 2, repeated=True)