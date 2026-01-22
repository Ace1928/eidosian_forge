from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesCloudschedulersourcesReplaceCloudSchedulerSourceRequest(_messages.Message):
    """A AnthoseventsNamespacesCloudschedulersourcesReplaceCloudSchedulerSource
  Request object.

  Fields:
    cloudSchedulerSource: A CloudSchedulerSource resource to be passed as the
      request body.
    name: The name of the cloudschedulersource being retrieved. If needed,
      replace {namespace_id} with the project ID.
  """
    cloudSchedulerSource = _messages.MessageField('CloudSchedulerSource', 1)
    name = _messages.StringField(2, required=True)