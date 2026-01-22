from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualityScanRuleResult(_messages.Message):
    """Information about the result of a data quality rule for data quality
  scan. The monitored resource is 'DataScan'.

  Enums:
    EvalutionTypeValueValuesEnum: The evaluation type of the data quality
      rule.
    ResultValueValuesEnum: The result of the data quality rule.
    RuleTypeValueValuesEnum: The type of the data quality rule.

  Fields:
    column: The column which this rule is evaluated against.
    dataSource: The data source of the data scan (e.g. BigQuery table name).
    evaluatedRowCount: The number of rows evaluated against the data quality
      rule. This field is only valid for rules of PER_ROW evaluation type.
    evalutionType: The evaluation type of the data quality rule.
    jobId: Identifier of the specific data scan job this log entry is for.
    nullRowCount: The number of rows with null values in the specified column.
    passedRowCount: The number of rows which passed a rule evaluation. This
      field is only valid for rules of PER_ROW evaluation type.
    result: The result of the data quality rule.
    ruleDimension: The dimension of the data quality rule.
    ruleName: The name of the data quality rule.
    ruleType: The type of the data quality rule.
    thresholdPercent: The passing threshold (0.0, 100.0) of the data quality
      rule.
  """

    class EvalutionTypeValueValuesEnum(_messages.Enum):
        """The evaluation type of the data quality rule.

    Values:
      EVALUATION_TYPE_UNSPECIFIED: An unspecified evaluation type.
      PER_ROW: The rule evaluation is done at per row level.
      AGGREGATE: The rule evaluation is done for an aggregate of rows.
    """
        EVALUATION_TYPE_UNSPECIFIED = 0
        PER_ROW = 1
        AGGREGATE = 2

    class ResultValueValuesEnum(_messages.Enum):
        """The result of the data quality rule.

    Values:
      RESULT_UNSPECIFIED: An unspecified result.
      PASSED: The data quality rule passed.
      FAILED: The data quality rule failed.
    """
        RESULT_UNSPECIFIED = 0
        PASSED = 1
        FAILED = 2

    class RuleTypeValueValuesEnum(_messages.Enum):
        """The type of the data quality rule.

    Values:
      RULE_TYPE_UNSPECIFIED: An unspecified rule type.
      NON_NULL_EXPECTATION: Please see https://cloud.google.com/dataplex/docs/
        reference/rest/v1/DataQualityRule#nonnullexpectation.
      RANGE_EXPECTATION: Please see https://cloud.google.com/dataplex/docs/ref
        erence/rest/v1/DataQualityRule#rangeexpectation.
      REGEX_EXPECTATION: Please see https://cloud.google.com/dataplex/docs/ref
        erence/rest/v1/DataQualityRule#regexexpectation.
      ROW_CONDITION_EXPECTATION: Please see https://cloud.google.com/dataplex/
        docs/reference/rest/v1/DataQualityRule#rowconditionexpectation.
      SET_EXPECTATION: Please see https://cloud.google.com/dataplex/docs/refer
        ence/rest/v1/DataQualityRule#setexpectation.
      STATISTIC_RANGE_EXPECTATION: Please see https://cloud.google.com/dataple
        x/docs/reference/rest/v1/DataQualityRule#statisticrangeexpectation.
      TABLE_CONDITION_EXPECTATION: Please see https://cloud.google.com/dataple
        x/docs/reference/rest/v1/DataQualityRule#tableconditionexpectation.
      UNIQUENESS_EXPECTATION: Please see https://cloud.google.com/dataplex/doc
        s/reference/rest/v1/DataQualityRule#uniquenessexpectation.
    """
        RULE_TYPE_UNSPECIFIED = 0
        NON_NULL_EXPECTATION = 1
        RANGE_EXPECTATION = 2
        REGEX_EXPECTATION = 3
        ROW_CONDITION_EXPECTATION = 4
        SET_EXPECTATION = 5
        STATISTIC_RANGE_EXPECTATION = 6
        TABLE_CONDITION_EXPECTATION = 7
        UNIQUENESS_EXPECTATION = 8
    column = _messages.StringField(1)
    dataSource = _messages.StringField(2)
    evaluatedRowCount = _messages.IntegerField(3)
    evalutionType = _messages.EnumField('EvalutionTypeValueValuesEnum', 4)
    jobId = _messages.StringField(5)
    nullRowCount = _messages.IntegerField(6)
    passedRowCount = _messages.IntegerField(7)
    result = _messages.EnumField('ResultValueValuesEnum', 8)
    ruleDimension = _messages.StringField(9)
    ruleName = _messages.StringField(10)
    ruleType = _messages.EnumField('RuleTypeValueValuesEnum', 11)
    thresholdPercent = _messages.FloatField(12)