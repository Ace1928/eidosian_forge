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
def _CreateAptSettings(args, messages):
    """Creates an AptSettings message from input arguments."""
    if not any([args.apt_dist, args.apt_excludes, args.apt_exclusive_packages]):
        return None
    return messages.AptSettings(type=messages.AptSettings.TypeValueValuesEnum.DIST if args.apt_dist else messages.AptSettings.TypeValueValuesEnum.TYPE_UNSPECIFIED, excludes=args.apt_excludes if args.apt_excludes else [], exclusivePackages=args.apt_exclusive_packages if args.apt_exclusive_packages else [])