from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1DoubleArray(_messages.Message):
    """A list of double values.

  Fields:
    values: A list of double values.
  """
    values = _messages.FloatField(1, repeated=True)