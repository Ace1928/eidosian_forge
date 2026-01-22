from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAutoscalingPoliciesResponse(_messages.Message):
    """A response to a request to list autoscaling policies in a project.

  Fields:
    nextPageToken: Output only. This token is included in the response if
      there are more results to fetch.
    policies: Output only. Autoscaling policies list.
  """
    nextPageToken = _messages.StringField(1)
    policies = _messages.MessageField('AutoscalingPolicy', 2, repeated=True)