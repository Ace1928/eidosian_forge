from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1StudyConfigParameterSpecCategoricalValueSpec(_messages.Message):
    """A GoogleCloudMlV1StudyConfigParameterSpecCategoricalValueSpec object.

  Fields:
    values: Must be specified if type is `CATEGORICAL`. The list of possible
      categories.
  """
    values = _messages.StringField(1, repeated=True)