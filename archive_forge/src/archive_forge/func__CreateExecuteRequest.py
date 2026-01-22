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
def _CreateExecuteRequest(messages, project, description, dry_run, duration, patch_config, patch_rollout, display_name, filter_all, filter_group_labels, filter_zones, filter_names, filter_name_prefixes):
    """Creates an ExecuteRequest message for the Beta track."""
    patch_instance_filter = _CreatePatchInstanceFilter(messages, filter_all, filter_group_labels, filter_zones, filter_names, filter_name_prefixes)
    if patch_rollout:
        return messages.OsconfigProjectsPatchJobsExecuteRequest(executePatchJobRequest=messages.ExecutePatchJobRequest(description=description, displayName=display_name, dryRun=dry_run, duration=duration, instanceFilter=patch_instance_filter, patchConfig=patch_config, rollout=patch_rollout), parent=osconfig_command_utils.GetProjectUriPath(project))
    else:
        return messages.OsconfigProjectsPatchJobsExecuteRequest(executePatchJobRequest=messages.ExecutePatchJobRequest(description=description, displayName=display_name, dryRun=dry_run, duration=duration, instanceFilter=patch_instance_filter, patchConfig=patch_config), parent=osconfig_command_utils.GetProjectUriPath(project))