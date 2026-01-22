from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LanguageLabsAidaTrustRecitationProtoRecitationResult(_messages.Message):
    """The recitation result for one input

  Enums:
    RecitationActionValueValuesEnum: The recitation action for one given
      input. When its segments contain different actions, the overall action
      will be returned in the precedence of BLOCK > CITE > NO_ACTION. When the
      given input is not found in any source, the recitation action will not
      be specified.

  Fields:
    dynamicSegmentResults: A LanguageLabsAidaTrustRecitationProtoSegmentResult
      attribute.
    recitationAction: The recitation action for one given input. When its
      segments contain different actions, the overall action will be returned
      in the precedence of BLOCK > CITE > NO_ACTION. When the given input is
      not found in any source, the recitation action will not be specified.
    trainingSegmentResults: A
      LanguageLabsAidaTrustRecitationProtoSegmentResult attribute.
  """

    class RecitationActionValueValuesEnum(_messages.Enum):
        """The recitation action for one given input. When its segments contain
    different actions, the overall action will be returned in the precedence
    of BLOCK > CITE > NO_ACTION. When the given input is not found in any
    source, the recitation action will not be specified.

    Values:
      ACTION_UNSPECIFIED: <no description>
      CITE: indicate that attribution must be shown for a Segment
      BLOCK: indicate that a Segment should be blocked from being used
      NO_ACTION: for tagging high-frequency code snippets
      EXEMPT_FOUND_IN_PROMPT: The recitation was found in prompt and is
        exempted from overall results
    """
        ACTION_UNSPECIFIED = 0
        CITE = 1
        BLOCK = 2
        NO_ACTION = 3
        EXEMPT_FOUND_IN_PROMPT = 4
    dynamicSegmentResults = _messages.MessageField('LanguageLabsAidaTrustRecitationProtoSegmentResult', 1, repeated=True)
    recitationAction = _messages.EnumField('RecitationActionValueValuesEnum', 2)
    trainingSegmentResults = _messages.MessageField('LanguageLabsAidaTrustRecitationProtoSegmentResult', 3, repeated=True)