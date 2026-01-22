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
class WorkflowsClient(object):
    """Client for Workflows service in the Cloud Workflows API."""

    def __init__(self, api_version):
        self.client = apis.GetClientInstance('workflows', api_version)
        self.messages = self.client.MESSAGES_MODULE
        self._service = self.client.projects_locations_workflows

    def Get(self, workflow_ref):
        """Gets a Workflow.

    Args:
      workflow_ref: Resource reference to the Workflow to get.

    Returns:
      Workflow: The workflow if it exists, None otherwise.
    """
        get_req = self.messages.WorkflowsProjectsLocationsWorkflowsGetRequest(name=workflow_ref.RelativeName())
        try:
            return self._service.Get(get_req)
        except api_exceptions.HttpNotFoundError:
            return None

    def Create(self, workflow_ref, workflow):
        """Creates a Workflow.

    Args:
      workflow_ref: Resource reference to the Workflow to create.
      workflow: Workflow resource message to create.

    Returns:
      Long-running operation for create.
    """
        create_req = self.messages.WorkflowsProjectsLocationsWorkflowsCreateRequest(parent=workflow_ref.Parent().RelativeName(), workflow=workflow, workflowId=workflow_ref.Name())
        return self._service.Create(create_req)

    def Patch(self, workflow_ref, workflow, updated_fields):
        """Updates a Workflow.

    If updated fields are specified it uses patch semantics.

    Args:
      workflow_ref: Resource reference to the Workflow to update.
      workflow: Workflow resource message to update.
      updated_fields: List of the updated fields used in a patch request.

    Returns:
      Long-running operation for update.
    """
        update_mask = ','.join(sorted(updated_fields))
        patch_req = self.messages.WorkflowsProjectsLocationsWorkflowsPatchRequest(name=workflow_ref.RelativeName(), updateMask=update_mask, workflow=workflow)
        return self._service.Patch(patch_req)

    def BuildWorkflowFromArgs(self, args, old_workflow, release_track):
        """Creates a workflow from command-line arguments.

    Args:
      args: The arguments of the gcloud command.
      old_workflow: The workflow from previous revision.
      release_track: The gcloud release track used in the command.

    Returns:
      workflow: The consturcted Workflow message from the passed in arguments.
      updated_fields: The workflow fields that are updated.
    """
        workflow = self.messages.Workflow()
        updated_fields = []
        flags.SetSource(args, workflow, updated_fields)
        flags.SetDescription(args, workflow, updated_fields)
        flags.SetServiceAccount(args, workflow, updated_fields)
        labels = labels_util.ParseCreateArgs(args, self.messages.Workflow.LabelsValue)
        flags.SetLabels(labels, workflow, updated_fields)
        if release_track == base.ReleaseTrack.GA:
            flags.SetKmsKey(args, workflow, updated_fields)
            env_vars = None
            if args.IsSpecified('set_env_vars'):
                env_vars = labels_util.ParseCreateArgs(args, self.messages.Workflow.UserEnvVarsValue, 'set_env_vars')
            if args.IsSpecified('env_vars_file'):
                if len(args.env_vars_file) > flags.USER_ENV_VARS_LIMIT:
                    raise arg_parsers.ArgumentTypeError('too many environment variables, limit is {max_len}.'.format(max_len=flags.USER_ENV_VARS_LIMIT))
                env_vars = labels_util.ParseCreateArgs(args, self.messages.Workflow.UserEnvVarsValue, 'env_vars_file')
            if args.IsSpecified('clear_env_vars'):
                env_vars = flags.CLEAR_ENVIRONMENT
            flags.SetUserEnvVars(env_vars, workflow, updated_fields)
            env_vars = None
            if args.IsSpecified('update_env_vars'):
                env_vars = {p.key: p.value for p in old_workflow.userEnvVars.additionalProperties}
                env_vars.update(args.update_env_vars)
            if args.IsSpecified('remove_env_vars'):
                env_vars = {p.key: p.value for p in old_workflow.userEnvVars.additionalProperties}
                for v in args.remove_env_vars:
                    if v in env_vars:
                        del env_vars[v]
                    else:
                        raise arg_parsers.argparse.ArgumentError(argument=None, message='key {k} is not found.'.format(k=v))
            flags.UpdateUserEnvVars(env_vars, workflow, updated_fields)
            if args.IsSpecified('call_log_level'):
                call_log_level_enum = self.messages.Workflow.CallLogLevelValueValuesEnum
                log_level = arg_utils.ChoiceToEnum(args.call_log_level, call_log_level_enum, valid_choices=['none', 'log-all-calls', 'log-errors-only', 'log-none'])
                flags.SetWorkflowLoggingArg(log_level, workflow, updated_fields)
        return (workflow, updated_fields)

    def WaitForOperation(self, operation, workflow_ref):
        """Waits until the given long-running operation is complete."""
        operation_ref = resources.REGISTRY.Parse(operation.name, collection='workflows.projects.locations.operations')
        operations = poller_utils.OperationsClient(self.client, self.messages)
        poller = poller_utils.WorkflowsOperationPoller(workflows=self, operations=operations, workflow_ref=workflow_ref)
        progress_string = 'Waiting for operation [{}] to complete'.format(operation_ref.Name())
        return waiter.WaitFor(poller, operation_ref, progress_string)