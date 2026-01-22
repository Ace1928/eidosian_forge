from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EntityIssue(_messages.Message):
    """Issue related to the entity.

  Enums:
    EntityTypeValueValuesEnum: The entity type (if the DDL is for a sub
      entity).
    SeverityValueValuesEnum: Severity of the issue
    TypeValueValuesEnum: The type of the issue.

  Fields:
    code: Error/Warning code
    ddl: The ddl which caused the issue, if relevant.
    entityType: The entity type (if the DDL is for a sub entity).
    id: Unique Issue ID.
    message: Issue detailed message
    position: The position of the issue found, if relevant.
    severity: Severity of the issue
    type: The type of the issue.
  """

    class EntityTypeValueValuesEnum(_messages.Enum):
        """The entity type (if the DDL is for a sub entity).

    Values:
      DATABASE_ENTITY_TYPE_UNSPECIFIED: Unspecified database entity type.
      DATABASE_ENTITY_TYPE_SCHEMA: Schema.
      DATABASE_ENTITY_TYPE_TABLE: Table.
      DATABASE_ENTITY_TYPE_COLUMN: Column.
      DATABASE_ENTITY_TYPE_CONSTRAINT: Constraint.
      DATABASE_ENTITY_TYPE_INDEX: Index.
      DATABASE_ENTITY_TYPE_TRIGGER: Trigger.
      DATABASE_ENTITY_TYPE_VIEW: View.
      DATABASE_ENTITY_TYPE_SEQUENCE: Sequence.
      DATABASE_ENTITY_TYPE_STORED_PROCEDURE: Stored Procedure.
      DATABASE_ENTITY_TYPE_FUNCTION: Function.
      DATABASE_ENTITY_TYPE_SYNONYM: Synonym.
      DATABASE_ENTITY_TYPE_DATABASE_PACKAGE: Package.
      DATABASE_ENTITY_TYPE_UDT: UDT.
      DATABASE_ENTITY_TYPE_MATERIALIZED_VIEW: Materialized View.
      DATABASE_ENTITY_TYPE_DATABASE: Database.
    """
        DATABASE_ENTITY_TYPE_UNSPECIFIED = 0
        DATABASE_ENTITY_TYPE_SCHEMA = 1
        DATABASE_ENTITY_TYPE_TABLE = 2
        DATABASE_ENTITY_TYPE_COLUMN = 3
        DATABASE_ENTITY_TYPE_CONSTRAINT = 4
        DATABASE_ENTITY_TYPE_INDEX = 5
        DATABASE_ENTITY_TYPE_TRIGGER = 6
        DATABASE_ENTITY_TYPE_VIEW = 7
        DATABASE_ENTITY_TYPE_SEQUENCE = 8
        DATABASE_ENTITY_TYPE_STORED_PROCEDURE = 9
        DATABASE_ENTITY_TYPE_FUNCTION = 10
        DATABASE_ENTITY_TYPE_SYNONYM = 11
        DATABASE_ENTITY_TYPE_DATABASE_PACKAGE = 12
        DATABASE_ENTITY_TYPE_UDT = 13
        DATABASE_ENTITY_TYPE_MATERIALIZED_VIEW = 14
        DATABASE_ENTITY_TYPE_DATABASE = 15

    class SeverityValueValuesEnum(_messages.Enum):
        """Severity of the issue

    Values:
      ISSUE_SEVERITY_UNSPECIFIED: Unspecified issue severity
      ISSUE_SEVERITY_INFO: Info
      ISSUE_SEVERITY_WARNING: Warning
      ISSUE_SEVERITY_ERROR: Error
    """
        ISSUE_SEVERITY_UNSPECIFIED = 0
        ISSUE_SEVERITY_INFO = 1
        ISSUE_SEVERITY_WARNING = 2
        ISSUE_SEVERITY_ERROR = 3

    class TypeValueValuesEnum(_messages.Enum):
        """The type of the issue.

    Values:
      ISSUE_TYPE_UNSPECIFIED: Unspecified issue type.
      ISSUE_TYPE_DDL: Issue originated from the DDL
      ISSUE_TYPE_APPLY: Issue originated during the apply process
      ISSUE_TYPE_CONVERT: Issue originated during the convert process
    """
        ISSUE_TYPE_UNSPECIFIED = 0
        ISSUE_TYPE_DDL = 1
        ISSUE_TYPE_APPLY = 2
        ISSUE_TYPE_CONVERT = 3
    code = _messages.StringField(1)
    ddl = _messages.StringField(2)
    entityType = _messages.EnumField('EntityTypeValueValuesEnum', 3)
    id = _messages.StringField(4)
    message = _messages.StringField(5)
    position = _messages.MessageField('Position', 6)
    severity = _messages.EnumField('SeverityValueValuesEnum', 7)
    type = _messages.EnumField('TypeValueValuesEnum', 8)