from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1Scalar(_messages.Message):
    """One point viewable on a scalar metric plot.

  Fields:
    value: Value of the point at this step / timestamp.
  """
    value = _messages.FloatField(1)