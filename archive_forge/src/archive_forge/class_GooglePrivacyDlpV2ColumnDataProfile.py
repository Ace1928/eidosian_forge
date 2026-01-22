from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ColumnDataProfile(_messages.Message):
    """The profile for a scanned column within a table.

  Enums:
    ColumnTypeValueValuesEnum: The data type of a given column.
    EstimatedNullPercentageValueValuesEnum: Approximate percentage of entries
      being null in the column.
    EstimatedUniquenessScoreValueValuesEnum: Approximate uniqueness of the
      column.
    PolicyStateValueValuesEnum: Indicates if a policy tag has been applied to
      the column.
    StateValueValuesEnum: State of a profile.

  Fields:
    column: The name of the column.
    columnInfoType: If it's been determined this column can be identified as a
      single type, this will be set. Otherwise the column either has
      unidentifiable content or mixed types.
    columnType: The data type of a given column.
    dataRiskLevel: The data risk level for this column.
    datasetId: The BigQuery dataset ID.
    datasetLocation: The BigQuery location where the dataset's data is stored.
      See https://cloud.google.com/bigquery/docs/locations for supported
      locations.
    datasetProjectId: The Google Cloud project ID that owns the profiled
      resource.
    estimatedNullPercentage: Approximate percentage of entries being null in
      the column.
    estimatedUniquenessScore: Approximate uniqueness of the column.
    freeTextScore: The likelihood that this column contains free-form text. A
      value close to 1 may indicate the column is likely to contain free-form
      or natural language text. Range in 0-1.
    name: The name of the profile.
    otherMatches: Other types found within this column. List will be
      unordered.
    policyState: Indicates if a policy tag has been applied to the column.
    profileLastGenerated: The last time the profile was generated.
    profileStatus: Success or error status from the most recent profile
      generation attempt. May be empty if the profile is still being
      generated.
    sensitivityScore: The sensitivity of this column.
    state: State of a profile.
    tableDataProfile: The resource name of the table data profile.
    tableFullResource: The resource name of the resource this column is
      within.
    tableId: The BigQuery table ID.
  """

    class ColumnTypeValueValuesEnum(_messages.Enum):
        """The data type of a given column.

    Values:
      COLUMN_DATA_TYPE_UNSPECIFIED: Invalid type.
      TYPE_INT64: Encoded as a string in decimal format.
      TYPE_BOOL: Encoded as a boolean "false" or "true".
      TYPE_FLOAT64: Encoded as a number, or string "NaN", "Infinity" or
        "-Infinity".
      TYPE_STRING: Encoded as a string value.
      TYPE_BYTES: Encoded as a base64 string per RFC 4648, section 4.
      TYPE_TIMESTAMP: Encoded as an RFC 3339 timestamp with mandatory "Z" time
        zone string: 1985-04-12T23:20:50.52Z
      TYPE_DATE: Encoded as RFC 3339 full-date format string: 1985-04-12
      TYPE_TIME: Encoded as RFC 3339 partial-time format string: 23:20:50.52
      TYPE_DATETIME: Encoded as RFC 3339 full-date "T" partial-time:
        1985-04-12T23:20:50.52
      TYPE_GEOGRAPHY: Encoded as WKT
      TYPE_NUMERIC: Encoded as a decimal string.
      TYPE_RECORD: Container of ordered fields, each with a type and field
        name.
      TYPE_BIGNUMERIC: Decimal type.
      TYPE_JSON: Json type.
      TYPE_INTERVAL: Interval type.
      TYPE_RANGE_DATE: Range type.
      TYPE_RANGE_DATETIME: Range type.
      TYPE_RANGE_TIMESTAMP: Range type.
    """
        COLUMN_DATA_TYPE_UNSPECIFIED = 0
        TYPE_INT64 = 1
        TYPE_BOOL = 2
        TYPE_FLOAT64 = 3
        TYPE_STRING = 4
        TYPE_BYTES = 5
        TYPE_TIMESTAMP = 6
        TYPE_DATE = 7
        TYPE_TIME = 8
        TYPE_DATETIME = 9
        TYPE_GEOGRAPHY = 10
        TYPE_NUMERIC = 11
        TYPE_RECORD = 12
        TYPE_BIGNUMERIC = 13
        TYPE_JSON = 14
        TYPE_INTERVAL = 15
        TYPE_RANGE_DATE = 16
        TYPE_RANGE_DATETIME = 17
        TYPE_RANGE_TIMESTAMP = 18

    class EstimatedNullPercentageValueValuesEnum(_messages.Enum):
        """Approximate percentage of entries being null in the column.

    Values:
      NULL_PERCENTAGE_LEVEL_UNSPECIFIED: Unused.
      NULL_PERCENTAGE_VERY_LOW: Very few null entries.
      NULL_PERCENTAGE_LOW: Some null entries.
      NULL_PERCENTAGE_MEDIUM: A few null entries.
      NULL_PERCENTAGE_HIGH: A lot of null entries.
    """
        NULL_PERCENTAGE_LEVEL_UNSPECIFIED = 0
        NULL_PERCENTAGE_VERY_LOW = 1
        NULL_PERCENTAGE_LOW = 2
        NULL_PERCENTAGE_MEDIUM = 3
        NULL_PERCENTAGE_HIGH = 4

    class EstimatedUniquenessScoreValueValuesEnum(_messages.Enum):
        """Approximate uniqueness of the column.

    Values:
      UNIQUENESS_SCORE_LEVEL_UNSPECIFIED: Some columns do not have estimated
        uniqueness. Possible reasons include having too few values.
      UNIQUENESS_SCORE_LOW: Low uniqueness, possibly a boolean, enum or
        similiarly typed column.
      UNIQUENESS_SCORE_MEDIUM: Medium uniqueness.
      UNIQUENESS_SCORE_HIGH: High uniqueness, possibly a column of free text
        or unique identifiers.
    """
        UNIQUENESS_SCORE_LEVEL_UNSPECIFIED = 0
        UNIQUENESS_SCORE_LOW = 1
        UNIQUENESS_SCORE_MEDIUM = 2
        UNIQUENESS_SCORE_HIGH = 3

    class PolicyStateValueValuesEnum(_messages.Enum):
        """Indicates if a policy tag has been applied to the column.

    Values:
      COLUMN_POLICY_STATE_UNSPECIFIED: No policy tags.
      COLUMN_POLICY_TAGGED: Column has policy tag applied.
    """
        COLUMN_POLICY_STATE_UNSPECIFIED = 0
        COLUMN_POLICY_TAGGED = 1

    class StateValueValuesEnum(_messages.Enum):
        """State of a profile.

    Values:
      STATE_UNSPECIFIED: Unused.
      RUNNING: The profile is currently running. Once a profile has finished
        it will transition to DONE.
      DONE: The profile is no longer generating. If profile_status.status.code
        is 0, the profile succeeded, otherwise, it failed.
    """
        STATE_UNSPECIFIED = 0
        RUNNING = 1
        DONE = 2
    column = _messages.StringField(1)
    columnInfoType = _messages.MessageField('GooglePrivacyDlpV2InfoTypeSummary', 2)
    columnType = _messages.EnumField('ColumnTypeValueValuesEnum', 3)
    dataRiskLevel = _messages.MessageField('GooglePrivacyDlpV2DataRiskLevel', 4)
    datasetId = _messages.StringField(5)
    datasetLocation = _messages.StringField(6)
    datasetProjectId = _messages.StringField(7)
    estimatedNullPercentage = _messages.EnumField('EstimatedNullPercentageValueValuesEnum', 8)
    estimatedUniquenessScore = _messages.EnumField('EstimatedUniquenessScoreValueValuesEnum', 9)
    freeTextScore = _messages.FloatField(10)
    name = _messages.StringField(11)
    otherMatches = _messages.MessageField('GooglePrivacyDlpV2OtherInfoTypeSummary', 12, repeated=True)
    policyState = _messages.EnumField('PolicyStateValueValuesEnum', 13)
    profileLastGenerated = _messages.StringField(14)
    profileStatus = _messages.MessageField('GooglePrivacyDlpV2ProfileStatus', 15)
    sensitivityScore = _messages.MessageField('GooglePrivacyDlpV2SensitivityScore', 16)
    state = _messages.EnumField('StateValueValuesEnum', 17)
    tableDataProfile = _messages.StringField(18)
    tableFullResource = _messages.StringField(19)
    tableId = _messages.StringField(20)