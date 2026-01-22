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
def _AddCommonTopLevelArguments(parser):
    """Adds top-level argument flags for all tracks."""
    base.ASYNC_FLAG.AddToParser(parser)
    parser.add_argument('--description', type=str, help='Textual description of the patch job.')
    parser.add_argument('--display-name', type=str, help='Display name for this patch job. This does not have to be unique.')
    parser.add_argument('--dry-run', action='store_true', help='      Whether to execute this patch job as a dry run. If this patch job is a dry\n      run, instances are contacted, but the patch is not run.')
    parser.add_argument('--duration', type=arg_parsers.Duration(), help='      Total duration in which the patch job must complete. If the patch does not\n      complete in this time, the process times out. While some instances might\n      still be running the patch, they will not continue to work after\n      completing the current step. See $ gcloud topic datetimes for information\n      on specifying absolute time durations.\n\n      If unspecified, the job stays active until all instances complete the\n      patch.')
    base.ChoiceArgument('--reboot-config', help_str='Post-patch reboot settings.', choices={'default': "          The agent decides if a reboot is necessary by checking signals such as\n          registry keys or '/var/run/reboot-required'.", 'always': 'Always reboot the machine after the update completes.', 'never': 'Never reboot the machine after the update completes.'}).AddToParser(parser)