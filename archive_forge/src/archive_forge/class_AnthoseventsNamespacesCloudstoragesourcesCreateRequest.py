from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesCloudstoragesourcesCreateRequest(_messages.Message):
    """A AnthoseventsNamespacesCloudstoragesourcesCreateRequest object.

  Fields:
    cloudStorageSource: A CloudStorageSource resource to be passed as the
      request body.
    parent: The namespace in which this cloudstoragesource should be created.
  """
    cloudStorageSource = _messages.MessageField('CloudStorageSource', 1)
    parent = _messages.StringField(2, required=True)