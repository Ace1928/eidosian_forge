from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SafetySetting(_messages.Message):
    """Safety settings.

  Enums:
    CategoryValueValuesEnum: Required. Harm category.
    MethodValueValuesEnum: Optional. Specify if the threshold is used for
      probability or severity score. If not specified, the threshold is used
      for probability score.
    ThresholdValueValuesEnum: Required. The harm block threshold.

  Fields:
    category: Required. Harm category.
    method: Optional. Specify if the threshold is used for probability or
      severity score. If not specified, the threshold is used for probability
      score.
    threshold: Required. The harm block threshold.
  """

    class CategoryValueValuesEnum(_messages.Enum):
        """Required. Harm category.

    Values:
      HARM_CATEGORY_UNSPECIFIED: The harm category is unspecified.
      HARM_CATEGORY_HATE_SPEECH: The harm category is hate speech.
      HARM_CATEGORY_DANGEROUS_CONTENT: The harm category is dangerous content.
      HARM_CATEGORY_HARASSMENT: The harm category is harassment.
      HARM_CATEGORY_SEXUALLY_EXPLICIT: The harm category is sexually explicit
        content.
    """
        HARM_CATEGORY_UNSPECIFIED = 0
        HARM_CATEGORY_HATE_SPEECH = 1
        HARM_CATEGORY_DANGEROUS_CONTENT = 2
        HARM_CATEGORY_HARASSMENT = 3
        HARM_CATEGORY_SEXUALLY_EXPLICIT = 4

    class MethodValueValuesEnum(_messages.Enum):
        """Optional. Specify if the threshold is used for probability or severity
    score. If not specified, the threshold is used for probability score.

    Values:
      HARM_BLOCK_METHOD_UNSPECIFIED: The harm block method is unspecified.
      SEVERITY: The harm block method uses both probability and severity
        scores.
      PROBABILITY: The harm block method uses the probability score.
    """
        HARM_BLOCK_METHOD_UNSPECIFIED = 0
        SEVERITY = 1
        PROBABILITY = 2

    class ThresholdValueValuesEnum(_messages.Enum):
        """Required. The harm block threshold.

    Values:
      HARM_BLOCK_THRESHOLD_UNSPECIFIED: Unspecified harm block threshold.
      BLOCK_LOW_AND_ABOVE: Block low threshold and above (i.e. block more).
      BLOCK_MEDIUM_AND_ABOVE: Block medium threshold and above.
      BLOCK_ONLY_HIGH: Block only high threshold (i.e. block less).
      BLOCK_NONE: Block none.
    """
        HARM_BLOCK_THRESHOLD_UNSPECIFIED = 0
        BLOCK_LOW_AND_ABOVE = 1
        BLOCK_MEDIUM_AND_ABOVE = 2
        BLOCK_ONLY_HIGH = 3
        BLOCK_NONE = 4
    category = _messages.EnumField('CategoryValueValuesEnum', 1)
    method = _messages.EnumField('MethodValueValuesEnum', 2)
    threshold = _messages.EnumField('ThresholdValueValuesEnum', 3)