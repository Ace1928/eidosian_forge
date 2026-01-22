from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CustomInfoType(_messages.Message):
    """Custom information type provided by the user. Used to find domain-
  specific sensitive information configurable to the data in question.

  Enums:
    ExclusionTypeValueValuesEnum: If set to EXCLUSION_TYPE_EXCLUDE this
      infoType will not cause a finding to be returned. It still can be used
      for rules matching.
    LikelihoodValueValuesEnum: Likelihood to return for this CustomInfoType.
      This base value can be altered by a detection rule if the finding meets
      the criteria specified by the rule. Defaults to `VERY_LIKELY` if not
      specified.

  Fields:
    detectionRules: Set of detection rules to apply to all findings of this
      CustomInfoType. Rules are applied in order that they are specified. Not
      supported for the `surrogate_type` CustomInfoType.
    dictionary: A list of phrases to detect as a CustomInfoType.
    exclusionType: If set to EXCLUSION_TYPE_EXCLUDE this infoType will not
      cause a finding to be returned. It still can be used for rules matching.
    infoType: CustomInfoType can either be a new infoType, or an extension of
      built-in infoType, when the name matches one of existing infoTypes and
      that infoType is specified in `InspectContent.info_types` field.
      Specifying the latter adds findings to the one detected by the system.
      If built-in info type is not specified in `InspectContent.info_types`
      list then the name is treated as a custom info type.
    likelihood: Likelihood to return for this CustomInfoType. This base value
      can be altered by a detection rule if the finding meets the criteria
      specified by the rule. Defaults to `VERY_LIKELY` if not specified.
    regex: Regular expression based CustomInfoType.
    sensitivityScore: Sensitivity for this CustomInfoType. If this
      CustomInfoType extends an existing InfoType, the sensitivity here will
      take precedence over that of the original InfoType. If unset for a
      CustomInfoType, it will default to HIGH. This only applies to data
      profiling.
    storedType: Load an existing `StoredInfoType` resource for use in
      `InspectDataSource`. Not currently supported in `InspectContent`.
    surrogateType: Message for detecting output from deidentification
      transformations that support reversing.
  """

    class ExclusionTypeValueValuesEnum(_messages.Enum):
        """If set to EXCLUSION_TYPE_EXCLUDE this infoType will not cause a
    finding to be returned. It still can be used for rules matching.

    Values:
      EXCLUSION_TYPE_UNSPECIFIED: A finding of this custom info type will not
        be excluded from results.
      EXCLUSION_TYPE_EXCLUDE: A finding of this custom info type will be
        excluded from final results, but can still affect rule execution.
    """
        EXCLUSION_TYPE_UNSPECIFIED = 0
        EXCLUSION_TYPE_EXCLUDE = 1

    class LikelihoodValueValuesEnum(_messages.Enum):
        """Likelihood to return for this CustomInfoType. This base value can be
    altered by a detection rule if the finding meets the criteria specified by
    the rule. Defaults to `VERY_LIKELY` if not specified.

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
    detectionRules = _messages.MessageField('GooglePrivacyDlpV2DetectionRule', 1, repeated=True)
    dictionary = _messages.MessageField('GooglePrivacyDlpV2Dictionary', 2)
    exclusionType = _messages.EnumField('ExclusionTypeValueValuesEnum', 3)
    infoType = _messages.MessageField('GooglePrivacyDlpV2InfoType', 4)
    likelihood = _messages.EnumField('LikelihoodValueValuesEnum', 5)
    regex = _messages.MessageField('GooglePrivacyDlpV2Regex', 6)
    sensitivityScore = _messages.MessageField('GooglePrivacyDlpV2SensitivityScore', 7)
    storedType = _messages.MessageField('GooglePrivacyDlpV2StoredType', 8)
    surrogateType = _messages.MessageField('GooglePrivacyDlpV2SurrogateType', 9)