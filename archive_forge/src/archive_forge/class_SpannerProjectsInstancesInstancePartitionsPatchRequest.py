from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesInstancePartitionsPatchRequest(_messages.Message):
    """A SpannerProjectsInstancesInstancePartitionsPatchRequest object.

  Fields:
    name: Required. A unique identifier for the instance partition. Values are
      of the form `projects//instances//instancePartitions/a-z*[a-z0-9]`. The
      final segment of the name must be between 2 and 64 characters in length.
      An instance partition's name cannot be changed after the instance
      partition is created.
    updateInstancePartitionRequest: A UpdateInstancePartitionRequest resource
      to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateInstancePartitionRequest = _messages.MessageField('UpdateInstancePartitionRequest', 2)