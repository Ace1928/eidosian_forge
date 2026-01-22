from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2LikelihoodAdjustment(_messages.Message):
    """Message for specifying an adjustment to the likelihood of a finding as
  part of a detection rule.

  Enums:
    FixedLikelihoodValueValuesEnum: Set the likelihood of a finding to a fixed
      value.

  Fields:
    fixedLikelihood: Set the likelihood of a finding to a fixed value.
    relativeLikelihood: Increase or decrease the likelihood by the specified
      number of levels. For example, if a finding would be `POSSIBLE` without
      the detection rule and `relative_likelihood` is 1, then it is upgraded
      to `LIKELY`, while a value of -1 would downgrade it to `UNLIKELY`.
      Likelihood may never drop below `VERY_UNLIKELY` or exceed `VERY_LIKELY`,
      so applying an adjustment of 1 followed by an adjustment of -1 when base
      likelihood is `VERY_LIKELY` will result in a final likelihood of
      `LIKELY`.
  """

    class FixedLikelihoodValueValuesEnum(_messages.Enum):
        """Set the likelihood of a finding to a fixed value.

    Values:
      LIKELIHOOD_UNSPECIFIED: Default value; same as POSSIBLE.
      VERY_UNLIKELY: Highest chance of a false positive.
      UNLIKELY: High chance of a false positive.
      POSSIBLE: Some matching signals. The default value.
      LIKELY: Low chance of a false positive.
      VERY_LIKELY: Confidence level is high. Lowest chance of a false
        positive.
    """
        LIKELIHOOD_UNSPECIFIED = 0
        VERY_UNLIKELY = 1
        UNLIKELY = 2
        POSSIBLE = 3
        LIKELY = 4
        VERY_LIKELY = 5
    fixedLikelihood = _messages.EnumField('FixedLikelihoodValueValuesEnum', 1)
    relativeLikelihood = _messages.IntegerField(2, variant=_messages.Variant.INT32)