from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import retry
import six
def WaitForWorkflowTemplateOperation(dataproc, operation, timeout_s=None, poll_period_s=5):
    """Poll dataproc Operation until its status is done or timeout reached.

  Args:
    dataproc: wrapper for Dataproc messages, resources, and client
    operation: Operation, message of the operation to be polled.
    timeout_s: number, seconds to poll with retries before timing out.
    poll_period_s: number, delay in seconds between requests.

  Returns:
    Operation: the return value of the last successful operations.get
    request.

  Raises:
    OperationError: if the operation times out or finishes with an error.
  """
    request = dataproc.messages.DataprocProjectsRegionsOperationsGetRequest(name=operation.name)
    log.status.Print('Waiting on operation [{0}].'.format(operation.name))
    start_time = time.time()
    operations = {'createCluster': None, 'deleteCluster': None}
    status = {}
    errors = {}
    while timeout_s is None or timeout_s > time.time() - start_time:
        try:
            operation = dataproc.client.projects_regions_operations.Get(request)
            metadata = ParseOperationJsonMetadata(operation.metadata, dataproc.messages.WorkflowMetadata)
            PrintWorkflowMetadata(metadata, status, operations, errors)
            if operation.done:
                break
        except apitools_exceptions.HttpError as http_exception:
            if IsClientHttpException(http_exception):
                raise
        time.sleep(poll_period_s)
    metadata = ParseOperationJsonMetadata(operation.metadata, dataproc.messages.WorkflowMetadata)
    if not operation.done:
        raise exceptions.OperationTimeoutError('Operation [{0}] timed out.'.format(operation.name))
    elif operation.error:
        raise exceptions.OperationError('Operation [{0}] failed: {1}.'.format(operation.name, FormatRpcError(operation.error)))
    for op in ['createCluster', 'deleteCluster']:
        if op in operations and operations[op] is not None and operations[op].error:
            raise exceptions.OperationError('Operation [{0}] failed: {1}.'.format(operations[op].operationId, operations[op].error))
    log.info('Operation [%s] finished after %.3f seconds', operation.name, time.time() - start_time)
    return operation