from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p1beta1SafeSearchAnnotation(_messages.Message):
    """Set of features pertaining to the image, computed by computer vision
  methods over safe-search verticals (for example, adult, spoof, medical,
  violence).

  Enums:
    AdultValueValuesEnum: Represents the adult content likelihood for the
      image. Adult content may contain elements such as nudity, pornographic
      images or cartoons, or sexual activities.
    MedicalValueValuesEnum: Likelihood that this is a medical image.
    RacyValueValuesEnum: Likelihood that the request image contains racy
      content. Racy content may include (but is not limited to) skimpy or
      sheer clothing, strategically covered nudity, lewd or provocative poses,
      or close-ups of sensitive body areas.
    SpoofValueValuesEnum: Spoof likelihood. The likelihood that an
      modification was made to the image's canonical version to make it appear
      funny or offensive.
    ViolenceValueValuesEnum: Likelihood that this image contains violent
      content. Violent content may include death, serious harm, or injury to
      individuals or groups of individuals.

  Fields:
    adult: Represents the adult content likelihood for the image. Adult
      content may contain elements such as nudity, pornographic images or
      cartoons, or sexual activities.
    medical: Likelihood that this is a medical image.
    racy: Likelihood that the request image contains racy content. Racy
      content may include (but is not limited to) skimpy or sheer clothing,
      strategically covered nudity, lewd or provocative poses, or close-ups of
      sensitive body areas.
    spoof: Spoof likelihood. The likelihood that an modification was made to
      the image's canonical version to make it appear funny or offensive.
    violence: Likelihood that this image contains violent content. Violent
      content may include death, serious harm, or injury to individuals or
      groups of individuals.
  """

    class AdultValueValuesEnum(_messages.Enum):
        """Represents the adult content likelihood for the image. Adult content
    may contain elements such as nudity, pornographic images or cartoons, or
    sexual activities.

    Values:
      UNKNOWN: Unknown likelihood.
      VERY_UNLIKELY: It is very unlikely.
      UNLIKELY: It is unlikely.
      POSSIBLE: It is possible.
      LIKELY: It is likely.
      VERY_LIKELY: It is very likely.
    """
        UNKNOWN = 0
        VERY_UNLIKELY = 1
        UNLIKELY = 2
        POSSIBLE = 3
        LIKELY = 4
        VERY_LIKELY = 5

    class MedicalValueValuesEnum(_messages.Enum):
        """Likelihood that this is a medical image.

    Values:
      UNKNOWN: Unknown likelihood.
      VERY_UNLIKELY: It is very unlikely.
      UNLIKELY: It is unlikely.
      POSSIBLE: It is possible.
      LIKELY: It is likely.
      VERY_LIKELY: It is very likely.
    """
        UNKNOWN = 0
        VERY_UNLIKELY = 1
        UNLIKELY = 2
        POSSIBLE = 3
        LIKELY = 4
        VERY_LIKELY = 5

    class RacyValueValuesEnum(_messages.Enum):
        """Likelihood that the request image contains racy content. Racy content
    may include (but is not limited to) skimpy or sheer clothing,
    strategically covered nudity, lewd or provocative poses, or close-ups of
    sensitive body areas.

    Values:
      UNKNOWN: Unknown likelihood.
      VERY_UNLIKELY: It is very unlikely.
      UNLIKELY: It is unlikely.
      POSSIBLE: It is possible.
      LIKELY: It is likely.
      VERY_LIKELY: It is very likely.
    """
        UNKNOWN = 0
        VERY_UNLIKELY = 1
        UNLIKELY = 2
        POSSIBLE = 3
        LIKELY = 4
        VERY_LIKELY = 5

    class SpoofValueValuesEnum(_messages.Enum):
        """Spoof likelihood. The likelihood that an modification was made to the
    image's canonical version to make it appear funny or offensive.

    Values:
      UNKNOWN: Unknown likelihood.
      VERY_UNLIKELY: It is very unlikely.
      UNLIKELY: It is unlikely.
      POSSIBLE: It is possible.
      LIKELY: It is likely.
      VERY_LIKELY: It is very likely.
    """
        UNKNOWN = 0
        VERY_UNLIKELY = 1
        UNLIKELY = 2
        POSSIBLE = 3
        LIKELY = 4
        VERY_LIKELY = 5

    class ViolenceValueValuesEnum(_messages.Enum):
        """Likelihood that this image contains violent content. Violent content
    may include death, serious harm, or injury to individuals or groups of
    individuals.

    Values:
      UNKNOWN: Unknown likelihood.
      VERY_UNLIKELY: It is very unlikely.
      UNLIKELY: It is unlikely.
      POSSIBLE: It is possible.
      LIKELY: It is likely.
      VERY_LIKELY: It is very likely.
    """
        UNKNOWN = 0
        VERY_UNLIKELY = 1
        UNLIKELY = 2
        POSSIBLE = 3
        LIKELY = 4
        VERY_LIKELY = 5
    adult = _messages.EnumField('AdultValueValuesEnum', 1)
    medical = _messages.EnumField('MedicalValueValuesEnum', 2)
    racy = _messages.EnumField('RacyValueValuesEnum', 3)
    spoof = _messages.EnumField('SpoofValueValuesEnum', 4)
    violence = _messages.EnumField('ViolenceValueValuesEnum', 5)