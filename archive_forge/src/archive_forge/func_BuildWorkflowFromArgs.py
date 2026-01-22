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