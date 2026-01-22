from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SafetySetting(_messages.Message):
    """Safety setting, affecting the safety-blocking behavior. Passing a safety
  setting for a category changes the allowed proability that content is
  blocked.

  Enums:
    CategoryValueValuesEnum: Required. The category for this setting.
    ThresholdValueValuesEnum: Required. Controls the probability threshold at
      which harm is blocked.

  Fields:
    category: Required. The category for this setting.
    threshold: Required. Controls the probability threshold at which harm is
      blocked.
  """

    class CategoryValueValuesEnum(_messages.Enum):
        """Required. The category for this setting.

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

    class ThresholdValueValuesEnum(_messages.Enum):
        """Required. Controls the probability threshold at which harm is blocked.

    Values:
      HARM_BLOCK_THRESHOLD_UNSPECIFIED: Threshold is unspecified.
      BLOCK_LOW_AND_ABOVE: Content with NEGLIGIBLE will be allowed.
      BLOCK_MEDIUM_AND_ABOVE: Content with NEGLIGIBLE and LOW will be allowed.
      BLOCK_ONLY_HIGH: Content with NEGLIGIBLE, LOW, and MEDIUM will be
        allowed.
      BLOCK_NONE: All content will be allowed.
    """
        HARM_BLOCK_THRESHOLD_UNSPECIFIED = 0
        BLOCK_LOW_AND_ABOVE = 1
        BLOCK_MEDIUM_AND_ABOVE = 2
        BLOCK_ONLY_HIGH = 3
        BLOCK_NONE = 4
    category = _messages.EnumField('CategoryValueValuesEnum', 1)
    threshold = _messages.EnumField('ThresholdValueValuesEnum', 2)