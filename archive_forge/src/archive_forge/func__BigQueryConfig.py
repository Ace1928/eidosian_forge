from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions
def _BigQueryConfig(self, table, use_topic_schema, use_table_schema, write_metadata, drop_unknown_fields):
    """Builds BigQueryConfig message from argument values.

    Args:
      table (str): The name of the table
      use_topic_schema (bool): Whether or not to use the topic schema
      use_table_schema (bool): Whether or not to use the table schema
      write_metadata (bool): Whether or not to write metadata fields
      drop_unknown_fields (bool): Whether or not to drop fields that are only in
        the topic schema

    Returns:
      BigQueryConfig message or None
    """
    if table:
        return self.messages.BigQueryConfig(table=table, useTopicSchema=use_topic_schema, useTableSchema=use_table_schema, writeMetadata=write_metadata, dropUnknownFields=drop_unknown_fields)
    return None