from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LintResult(_messages.Message):
    """Structured response of a single validation unit.

  Enums:
    LevelValueValuesEnum: The validation unit level.
    SeverityValueValuesEnum: The validation unit severity.

  Fields:
    debugMessage: Human readable debug message associated with the issue.
    fieldName: The name of the field for which this lint result is about. For
      nested messages `field_name` consists of names of the embedded fields
      separated by period character. The top-level qualifier is the input
      object to lint in the request. For example, the `field_name` value
      `condition.expression` identifies a lint result for the `expression`
      field of the provided condition.
    level: The validation unit level.
    locationOffset: 0-based character position of problematic construct within
      the object identified by `field_name`. Currently, this is populated only
      for condition expression.
    severity: The validation unit severity.
    validationUnitName: The validation unit name, for instance
      "lintValidationUnits/ConditionComplexityCheck".
  """

    class LevelValueValuesEnum(_messages.Enum):
        """The validation unit level.

    Values:
      LEVEL_UNSPECIFIED: Level is unspecified.
      CONDITION: A validation unit which operates on an individual condition
        within a binding.
    """
        LEVEL_UNSPECIFIED = 0
        CONDITION = 1

    class SeverityValueValuesEnum(_messages.Enum):
        """The validation unit severity.

    Values:
      SEVERITY_UNSPECIFIED: Severity is unspecified.
      ERROR: A validation unit returns an error only for critical issues. If
        an attempt is made to set the problematic policy without rectifying
        the critical issue, it causes the `setPolicy` operation to fail.
      WARNING: Any issue which is severe enough but does not cause an error.
        For example, suspicious constructs in the input object will not
        necessarily fail `setPolicy`, but there is a high likelihood that they
        won't behave as expected during policy evaluation in `checkPolicy`.
        This includes the following common scenarios: - Unsatisfiable
        condition: Expired timestamp in date/time condition. - Ineffective
        condition: Condition on a pair which is granted unconditionally in
        another binding of the same policy.
      NOTICE: Reserved for the issues that are not severe as
        `ERROR`/`WARNING`, but need special handling. For instance, messages
        about skipped validation units are issued as `NOTICE`.
      INFO: Any informative statement which is not severe enough to raise
        `ERROR`/`WARNING`/`NOTICE`, like auto-correction recommendations on
        the input content. Note that current version of the linter does not
        utilize `INFO`.
      DEPRECATED: Deprecated severity level.
    """
        SEVERITY_UNSPECIFIED = 0
        ERROR = 1
        WARNING = 2
        NOTICE = 3
        INFO = 4
        DEPRECATED = 5
    debugMessage = _messages.StringField(1)
    fieldName = _messages.StringField(2)
    level = _messages.EnumField('LevelValueValuesEnum', 3)
    locationOffset = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    severity = _messages.EnumField('SeverityValueValuesEnum', 5)
    validationUnitName = _messages.StringField(6)