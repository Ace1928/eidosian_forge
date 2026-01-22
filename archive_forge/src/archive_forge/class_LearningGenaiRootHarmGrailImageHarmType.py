from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootHarmGrailImageHarmType(_messages.Message):
    """Harm type for images

  Enums:
    ImageHarmTypeValueListEntryValuesEnum:

  Fields:
    imageHarmType: A ImageHarmTypeValueListEntryValuesEnum attribute.
  """

    class ImageHarmTypeValueListEntryValuesEnum(_messages.Enum):
        """ImageHarmTypeValueListEntryValuesEnum enum type.

    Values:
      IMAGE_HARM_TYPE_UNSPECIFIED: <no description>
      IMAGE_HARM_TYPE_PORN: <no description>
      IMAGE_HARM_TYPE_VIOLENCE: <no description>
      IMAGE_HARM_TYPE_CSAI: <no description>
      IMAGE_HARM_TYPE_PEDO: <no description>
      IMAGE_HARM_TYPE_MINORS: <no description>
      IMAGE_HARM_TYPE_DANGEROUS: <no description>
      IMAGE_HARM_TYPE_MEDICAL: <no description>
      IMAGE_HARM_TYPE_RACY: <no description>
      IMAGE_HARM_TYPE_OBSCENE: <no description>
      IMAGE_HARM_TYPE_MINOR_PRESENCE: <no description>
      IMAGE_HARM_TYPE_GENERATIVE_MINOR_PRESENCE: <no description>
      IMAGE_HARM_TYPE_GENERATIVE_REALISTIC_VISIBLE_FACE: <no description>
    """
        IMAGE_HARM_TYPE_UNSPECIFIED = 0
        IMAGE_HARM_TYPE_PORN = 1
        IMAGE_HARM_TYPE_VIOLENCE = 2
        IMAGE_HARM_TYPE_CSAI = 3
        IMAGE_HARM_TYPE_PEDO = 4
        IMAGE_HARM_TYPE_MINORS = 5
        IMAGE_HARM_TYPE_DANGEROUS = 6
        IMAGE_HARM_TYPE_MEDICAL = 7
        IMAGE_HARM_TYPE_RACY = 8
        IMAGE_HARM_TYPE_OBSCENE = 9
        IMAGE_HARM_TYPE_MINOR_PRESENCE = 10
        IMAGE_HARM_TYPE_GENERATIVE_MINOR_PRESENCE = 11
        IMAGE_HARM_TYPE_GENERATIVE_REALISTIC_VISIBLE_FACE = 12
    imageHarmType = _messages.EnumField('ImageHarmTypeValueListEntryValuesEnum', 1, repeated=True)