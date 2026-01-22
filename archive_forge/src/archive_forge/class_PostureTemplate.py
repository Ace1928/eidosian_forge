from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PostureTemplate(_messages.Message):
    """========================== PostureTemplates ==========================
  Message describing PostureTemplate object.

  Enums:
    CategoriesValueListEntryValuesEnum:
    StateValueValuesEnum: Output only. State of PostureTemplate resource.

  Fields:
    categories: Output only. List of categories associated with a
      PostureTemplate. Based on it's associated policies we define the
      category, hence it is OUTPUT_ONLY field.
    description: Output only. Description of the Posture template.
    name: Output only. Identifier. The name of the Posture template will be of
      the format organizations/{organization}/locations/{location}/postureTemp
      lates/{postureTemplate}
    policySets: Output only. Policy_sets to be used by the user.
    revisionId: Output only. The revision_id of a PostureTemplate.
    state: Output only. State of PostureTemplate resource.
  """

    class CategoriesValueListEntryValuesEnum(_messages.Enum):
        """CategoriesValueListEntryValuesEnum enum type.

    Values:
      CATEGORY_UNSPECIFIED: Unspecified Category.
      AI: AI Category.
      AWS: Posture contains AWS policies.
      GCP: Posture contains GCP policies.
    """
        CATEGORY_UNSPECIFIED = 0
        AI = 1
        AWS = 2
        GCP = 3

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of PostureTemplate resource.

    Values:
      STATE_UNSPECIFIED: Unspecified state
      ACTIVE: If the Posture template is adhering to the latest controls and
        standards.
      DEPRECATED: If the Posture template controls and standards are outdated
        and not recommended for use.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        DEPRECATED = 2
    categories = _messages.EnumField('CategoriesValueListEntryValuesEnum', 1, repeated=True)
    description = _messages.StringField(2)
    name = _messages.StringField(3)
    policySets = _messages.MessageField('PolicySet', 4, repeated=True)
    revisionId = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)