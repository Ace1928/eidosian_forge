from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_projector
import six
def _CreateExecuteRequestAlpha(messages, project, description, dry_run, duration, patch_config, patch_rollout, display_name, filter_all, filter_group_labels, filter_zones, filter_names, filter_name_prefixes, filter_expression):
    """Creates an ExecuteRequest message for the Alpha track."""
    if filter_expression:
        return messages.OsconfigProjectsPatchJobsExecuteRequest(executePatchJobRequest=messages.ExecutePatchJobRequest(description=description, displayName=display_name, dryRun=dry_run, duration=duration, filter=filter_expression, patchConfig=patch_config, rollout=patch_rollout), parent=osconfig_command_utils.GetProjectUriPath(project))
    elif not any([filter_all, filter_group_labels, filter_zones, filter_names, filter_name_prefixes]):
        return messages.OsconfigProjectsPatchJobsExecuteRequest(executePatchJobRequest=messages.ExecutePatchJobRequest(description=description, displayName=display_name, dryRun=dry_run, duration=duration, instanceFilter=messages.PatchInstanceFilter(all=True), patchConfig=patch_config, rollout=patch_rollout), parent=osconfig_command_utils.GetProjectUriPath(project))
    else:
        return _CreateExecuteRequest(messages, project, description, dry_run, duration, patch_config, patch_rollout, display_name, filter_all, filter_group_labels, filter_zones, filter_names, filter_name_prefixes)