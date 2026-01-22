from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.api_lib.workflows import cache
from googlecloudsdk.api_lib.workflows import poller_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.workflows import flags
from googlecloudsdk.core import resources
class WorkflowExecutionClient(object):
    """Client for Workflows Execution service in the Cloud Workflows Execution API."""

    def __init__(self, api_version):
        self.client = apis.GetClientInstance('workflowexecutions', api_version)
        self.messages = self.client.MESSAGES_MODULE
        self._service = self.client.projects_locations_workflows_executions

    def Create(self, workflow_ref, data, call_log_level=None, labels=None, overflow_buffering_disabled=False):
        """Creates a Workflow execution.

    Args:
      workflow_ref: Resource reference to the Workflow to execute.
      data: Argments to use for executing the workflow.
      call_log_level: Level of call logging to apply during execution.
      labels: Labels associated to the execution.
      overflow_buffering_disabled: If set to true, the execution will not be
        backlogged when the concurrency quota is exhausted. Backlogged
        executions start when the concurrency quota becomes available.

    Returns:
      Execution: The workflow execution.
    """
        execution = self.messages.Execution()
        execution.argument = data
        if overflow_buffering_disabled:
            execution.disableConcurrencyQuotaOverflowBuffering = True
        if labels is not None:
            execution.labels = labels
        if call_log_level is not None and call_log_level != 'none':
            call_log_level_enum = self.messages.Execution.CallLogLevelValueValuesEnum
            execution.callLogLevel = arg_utils.ChoiceToEnum(call_log_level, call_log_level_enum, valid_choices=['none', 'log-all-calls', 'log-errors-only', 'log-none'])
        create_req = self.messages.WorkflowexecutionsProjectsLocationsWorkflowsExecutionsCreateRequest(parent=workflow_ref.RelativeName(), execution=execution)
        try:
            return self._service.Create(create_req)
        except api_exceptions.HttpError as e:
            raise exceptions.HttpException(e, error_format='{message}')

    def Get(self, execution_ref):
        """Gets a workflow execution.

    Args:
      execution_ref: Resource reference to the Workflow execution to get.

    Returns:
      Workflow: The workflow execution if it exists, an error exception
      otherwise.
    """
        if execution_ref is None:
            execution_ref = cache.get_cached_execution_id()
        get_req = self.messages.WorkflowexecutionsProjectsLocationsWorkflowsExecutionsGetRequest(name=execution_ref.RelativeName())
        try:
            return self._service.Get(get_req)
        except api_exceptions.HttpError as e:
            raise exceptions.HttpException(e, error_format='{message}')

    def WaitForExecution(self, execution_ref):
        """Waits until the given execution is complete or the maximum wait time is reached."""
        if execution_ref is None:
            execution_ref = cache.get_cached_execution_id()
        poller = poller_utils.ExecutionsPoller(workflow_execution=self)
        progress_string = 'Waiting for execution [{}] to complete'.format(execution_ref.Name())
        try:
            return waiter.WaitFor(poller, execution_ref, progress_string, pre_start_sleep_ms=100, max_wait_ms=86400000, exponential_sleep_multiplier=1.25, wait_ceiling_ms=60000)
        except waiter.TimeoutError:
            raise waiter.TimeoutError('Execution {0} has not finished in 24 hours. {1}'.format(execution_ref, _TIMEOUT_MESSAGE))
        except waiter.AbortWaitError:
            raise waiter.AbortWaitError('Aborting wait for execution {0}.'.format(execution_ref))