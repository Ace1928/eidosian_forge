from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1alpha2OperationGroup(_messages.Message):
    """Group of operations that need to be performed atomically.

  Fields:
    operations: List of operations across one or more resources that belong to
      this group. Loosely based on RFC6902 and should be performed in the
      order they appear.
  """
    operations = _messages.MessageField('GoogleCloudRecommenderV1alpha2Operation', 1, repeated=True)