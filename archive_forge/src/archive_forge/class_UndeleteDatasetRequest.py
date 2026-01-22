from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UndeleteDatasetRequest(_messages.Message):
    """Request format for undeleting a dataset.

  Fields:
    deletionTime: Optional. The exact time when the dataset was deleted. If
      not specified, the most recently deleted version is undeleted.
  """
    deletionTime = _messages.StringField(1)