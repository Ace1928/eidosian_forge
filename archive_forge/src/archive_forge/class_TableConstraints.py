from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TableConstraints(_messages.Message):
    """The TableConstraints defines the primary key and foreign key.

  Messages:
    ForeignKeysValueListEntry: Represents a foreign key constraint on a
      table's columns.
    PrimaryKeyValue: Represents the primary key constraint on a table's
      columns.

  Fields:
    foreignKeys: Optional. Present only if the table has a foreign key. The
      foreign key is not enforced.
    primaryKey: Represents the primary key constraint on a table's columns.
  """

    class ForeignKeysValueListEntry(_messages.Message):
        """Represents a foreign key constraint on a table's columns.

    Messages:
      ColumnReferencesValueListEntry: The pair of the foreign key column and
        primary key column.
      ReferencedTableValue: A ReferencedTableValue object.

    Fields:
      columnReferences: Required. The columns that compose the foreign key.
      name: Optional. Set only if the foreign key constraint is named.
      referencedTable: A ReferencedTableValue attribute.
    """

        class ColumnReferencesValueListEntry(_messages.Message):
            """The pair of the foreign key column and primary key column.

      Fields:
        referencedColumn: Required. The column in the primary key that are
          referenced by the referencing_column.
        referencingColumn: Required. The column that composes the foreign key.
      """
            referencedColumn = _messages.StringField(1)
            referencingColumn = _messages.StringField(2)

        class ReferencedTableValue(_messages.Message):
            """A ReferencedTableValue object.

      Fields:
        datasetId: A string attribute.
        projectId: A string attribute.
        tableId: A string attribute.
      """
            datasetId = _messages.StringField(1)
            projectId = _messages.StringField(2)
            tableId = _messages.StringField(3)
        columnReferences = _messages.MessageField('ColumnReferencesValueListEntry', 1, repeated=True)
        name = _messages.StringField(2)
        referencedTable = _messages.MessageField('ReferencedTableValue', 3)

    class PrimaryKeyValue(_messages.Message):
        """Represents the primary key constraint on a table's columns.

    Fields:
      columns: Required. The columns that are composed of the primary key
        constraint.
    """
        columns = _messages.StringField(1, repeated=True)
    foreignKeys = _messages.MessageField('ForeignKeysValueListEntry', 1, repeated=True)
    primaryKey = _messages.MessageField('PrimaryKeyValue', 2)