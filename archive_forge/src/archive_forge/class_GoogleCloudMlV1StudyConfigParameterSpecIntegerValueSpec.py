from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1StudyConfigParameterSpecIntegerValueSpec(_messages.Message):
    """A GoogleCloudMlV1StudyConfigParameterSpecIntegerValueSpec object.

  Fields:
    maxValue: Must be specified if type is `INTEGER`. Maximum value of the
      parameter.
    minValue: Must be specified if type is `INTEGER`. Minimum value of the
      parameter.
  """
    maxValue = _messages.IntegerField(1)
    minValue = _messages.IntegerField(2)