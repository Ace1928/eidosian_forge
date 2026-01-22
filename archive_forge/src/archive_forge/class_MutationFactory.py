from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from apitools.base.py import http_wrapper
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.spanner.sql import QueryHasDml
class MutationFactory(object):
    """Factory that creates and returns a mutation object in Cloud Spanner.

  A Mutation represents a sequence of inserts, updates and deletes that can be
  applied to rows and tables in a Cloud Spanner database.
  """
    msgs = apis.GetMessagesModule('spanner', 'v1')

    @classmethod
    def Insert(cls, table, data):
        """Constructs an INSERT mutation, which inserts a new row in a table.

    Args:
      table: String, the name of the table.
      data: A collections.OrderedDict, the keys of which are the column names
        and values are the column values to be inserted.

    Returns:
      An insert mutation operation.
    """
        return cls.msgs.Mutation(insert=cls._GetWrite(table, data))

    @classmethod
    def Update(cls, table, data):
        """Constructs an UPDATE mutation, which updates a row in a table.

    Args:
      table: String, the name of the table.
      data: An ordered dictionary where the keys are the column names and values
        are the column values to be updated.

    Returns:
      An update mutation operation.
    """
        return cls.msgs.Mutation(update=cls._GetWrite(table, data))

    @classmethod
    def Delete(cls, table, keys):
        """Constructs a DELETE mutation, which deletes a row in a table.

    Args:
      table: String, the name of the table.
      keys: String list, the primary key values of the row to delete.

    Returns:
      A delete mutation operation.
    """
        return cls.msgs.Mutation(delete=cls._GetDelete(table, keys))

    @classmethod
    def _GetWrite(cls, table, data):
        """Constructs Write object, which is needed for insert/update operations."""

        def _ToJson(msg):
            return extra_types.JsonProtoEncoder(extra_types.JsonArray(entries=msg.entry))
        encoding.RegisterCustomMessageCodec(encoder=_ToJson, decoder=None)(cls.msgs.Write.ValuesValueListEntry)
        json_columns = table.GetJsonData(data)
        json_column_names = [col.col_name for col in json_columns]
        json_column_values = [col.col_value for col in json_columns]
        return cls.msgs.Write(columns=json_column_names, table=table.name, values=[cls.msgs.Write.ValuesValueListEntry(entry=json_column_values)])

    @classmethod
    def _GetDelete(cls, table, keys):
        """Constructs Delete object, which is needed for delete operation."""

        def _ToJson(msg):
            return extra_types.JsonProtoEncoder(extra_types.JsonArray(entries=msg.entry))
        encoding.RegisterCustomMessageCodec(encoder=_ToJson, decoder=None)(cls.msgs.KeySet.KeysValueListEntry)
        key_set = cls.msgs.KeySet(keys=[cls.msgs.KeySet.KeysValueListEntry(entry=table.GetJsonKeys(keys))])
        return cls.msgs.Delete(table=table.name, keySet=key_set)