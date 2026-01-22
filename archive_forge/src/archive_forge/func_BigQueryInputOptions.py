from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def BigQueryInputOptions(table_name):
    """Convert BigQuery table name into GooglePrivacyDlpV2BigQueryOptions.

  Creates BigQuery input options for a job trigger.

  Args:
    table_name: str, BigQuery table name to create input options from in the
      form `<project_id>.<dataset_id>.<table_id>`.

  Returns:
    GooglePrivacyDlpV2BigQueryOptions, input options for job trigger.

  Raises:
    BigQueryTableNameError if table_name is improperly formatted.
  """
    project_id, data_set_id, table_id = _ValidateAndParseInputTableName(table_name)
    big_query_options = _GetMessageClass('GooglePrivacyDlpV2BigQueryOptions')
    big_query_table = _GetMessageClass('GooglePrivacyDlpV2BigQueryTable')
    table = big_query_table(datasetId=data_set_id, projectId=project_id, tableId=table_id)
    options = big_query_options(tableReference=table)
    return options