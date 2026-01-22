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
def _CreateWindowsUpdateSettings(args, messages):
    """Creates a WindowsUpdateSettings message from input arguments."""
    if not any([args.windows_classifications, args.windows_excludes, args.windows_exclusive_patches]):
        return None
    enums = messages.WindowsUpdateSettings.ClassificationsValueListEntryValuesEnum
    classifications = [arg_utils.ChoiceToEnum(c, enums) for c in args.windows_classifications] if args.windows_classifications else []
    return messages.WindowsUpdateSettings(classifications=classifications, excludes=args.windows_excludes if args.windows_excludes else [], exclusivePatches=args.windows_exclusive_patches if args.windows_exclusive_patches else [])