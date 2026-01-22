from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1BoolArray(_messages.Message):
    """A list of boolean values.

  Fields:
    values: A list of bool values.
  """
    values = _messages.BooleanField(1, repeated=True)