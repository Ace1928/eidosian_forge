from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Finding(_messages.Message):
    """Represents a piece of potentially sensitive content.

  Enums:
    LikelihoodValueValuesEnum: Confidence of how likely it is that the
      `info_type` is correct.

  Messages:
    LabelsValue: The labels associated with this `Finding`. Label keys must be
      between 1 and 63 characters long and must conform to the following
      regular expression: `[a-z]([-a-z0-9]*[a-z0-9])?`. Label values must be
      between 0 and 63 characters long and must conform to the regular
      expression `([a-z]([-a-z0-9]*[a-z0-9])?)?`. No more than 10 labels can
      be associated with a given finding. Examples: * `"environment" :
      "production"` * `"pipeline" : "etl"`

  Fields:
    createTime: Timestamp when finding was detected.
    findingId: The unique finding id.
    infoType: The type of content that might have been found. Provided if
      `excluded_types` is false.
    jobCreateTime: Time the job started that produced this finding.
    jobName: The job that stored the finding.
    labels: The labels associated with this `Finding`. Label keys must be
      between 1 and 63 characters long and must conform to the following
      regular expression: `[a-z]([-a-z0-9]*[a-z0-9])?`. Label values must be
      between 0 and 63 characters long and must conform to the regular
      expression `([a-z]([-a-z0-9]*[a-z0-9])?)?`. No more than 10 labels can
      be associated with a given finding. Examples: * `"environment" :
      "production"` * `"pipeline" : "etl"`
    likelihood: Confidence of how likely it is that the `info_type` is
      correct.
    location: Where the content was found.
    name: Resource name in format
      projects/{project}/locations/{location}/findings/{finding} Populated
      only when viewing persisted findings.
    quote: The content that was found. Even if the content is not textual, it
      may be converted to a textual representation here. Provided if
      `include_quote` is true and the finding is less than or equal to 4096
      bytes long. If the finding exceeds 4096 bytes in length, the quote may
      be omitted.
    quoteInfo: Contains data parsed from quotes. Only populated if
      include_quote was set to true and a supported infoType was requested.
      Currently supported infoTypes: DATE, DATE_OF_BIRTH and TIME.
    resourceName: The job that stored the finding.
    triggerName: Job trigger name, if applicable, for this finding.
  """

    class LikelihoodValueValuesEnum(_messages.Enum):
        """Confidence of how likely it is that the `info_type` is correct.

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels associated with this `Finding`. Label keys must be between
    1 and 63 characters long and must conform to the following regular
    expression: `[a-z]([-a-z0-9]*[a-z0-9])?`. Label values must be between 0
    and 63 characters long and must conform to the regular expression
    `([a-z]([-a-z0-9]*[a-z0-9])?)?`. No more than 10 labels can be associated
    with a given finding. Examples: * `"environment" : "production"` *
    `"pipeline" : "etl"`

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    findingId = _messages.StringField(2)
    infoType = _messages.MessageField('GooglePrivacyDlpV2InfoType', 3)
    jobCreateTime = _messages.StringField(4)
    jobName = _messages.StringField(5)
    labels = _messages.MessageField('LabelsValue', 6)
    likelihood = _messages.EnumField('LikelihoodValueValuesEnum', 7)
    location = _messages.MessageField('GooglePrivacyDlpV2Location', 8)
    name = _messages.StringField(9)
    quote = _messages.StringField(10)
    quoteInfo = _messages.MessageField('GooglePrivacyDlpV2QuoteInfo', 11)
    resourceName = _messages.StringField(12)
    triggerName = _messages.StringField(13)