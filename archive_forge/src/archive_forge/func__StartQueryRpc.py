from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional
import uuid
from absl import flags
from google.api_core.iam import Policy
from googleapiclient import http as http_request
import inflection
from clients import bigquery_client
from clients import client_dataset
from clients import client_reservation
from clients import table_reader as bq_table_reader
from clients import utils as bq_client_utils
from utils import bq_api_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
def _StartQueryRpc(self, query, dry_run=None, use_cache=None, preserve_nulls=None, request_id=None, maximum_bytes_billed=None, max_results=None, timeout_ms=None, min_completion_ratio=None, project_id=None, external_table_definitions_json=None, udf_resources=None, use_legacy_sql=None, location=None, connection_properties=None, **kwds):
    """Executes the given query using the rpc-style query api.

    Args:
      query: Query to execute.
      dry_run: Optional. Indicates whether the query will only be validated and
        return processing statistics instead of actually running.
      use_cache: Optional. Whether to use the query cache. Caching is
        best-effort only and you should not make assumptions about whether or
        how long a query result will be cached.
      preserve_nulls: Optional. Indicates whether to preserve nulls in input
        data. Temporary flag; will be removed in a future version.
      request_id: Optional. The idempotency token for jobs.query
      maximum_bytes_billed: Optional. Upper limit on the number of billed bytes.
      max_results: Maximum number of results to return.
      timeout_ms: Timeout, in milliseconds, for the call to query().
      min_completion_ratio: Optional. Specifies the minimum fraction of data
        that must be scanned before a query returns. This value should be
        between 0.0 and 1.0 inclusive.
      project_id: Project id to use.
      external_table_definitions_json: Json representation of external table
        definitions.
      udf_resources: Array of inline and external UDF code resources.
      use_legacy_sql: The choice of using Legacy SQL for the query is optional.
        If not specified, the server will automatically determine the dialect
        based on query information, such as dialect prefixes. If no prefixes are
        found, it will default to Legacy SQL.
      location: Optional. The geographic location where the job should run.
      connection_properties: Optional. Connection properties to use when running
        the query, presented as a list of key/value pairs. A key of "time_zone"
        indicates that the query will be run with the default timezone
        corresponding to the value.
      **kwds: Extra keyword arguments passed directly to jobs.Query().

    Returns:
      The query response.

    Raises:
      bq_error.BigqueryClientConfigurationError: if project_id and
        self.project_id are None.
      bq_error.BigqueryError: if query execution fails.
    """
    project_id = project_id or self.project_id
    if not project_id:
        raise bq_error.BigqueryClientConfigurationError('Cannot run a query without a project id.')
    request = {'query': query}
    if external_table_definitions_json:
        request['tableDefinitions'] = external_table_definitions_json
    if udf_resources:
        request['userDefinedFunctionResources'] = udf_resources
    if self.dataset_id:
        request['defaultDataset'] = bq_client_utils.GetQueryDefaultDataset(self.dataset_id)
    if request_id is None and flags.FLAGS.jobs_query_use_request_id:
        request_id = str(uuid.uuid4())
    bq_processor_utils.ApplyParameters(request, preserve_nulls=preserve_nulls, request_id=request_id, maximum_bytes_billed=maximum_bytes_billed, use_query_cache=use_cache, timeout_ms=timeout_ms, max_results=max_results, use_legacy_sql=use_legacy_sql, min_completion_ratio=min_completion_ratio, location=location)
    bq_processor_utils.ApplyParameters(request, connection_properties=connection_properties)
    bq_processor_utils.ApplyParameters(request, dry_run=dry_run)
    logging.debug('Calling self.apiclient.jobs().query(%s, %s, %s)', request, project_id, kwds)
    return self.apiclient.jobs().query(body=request, projectId=project_id, **kwds).execute()