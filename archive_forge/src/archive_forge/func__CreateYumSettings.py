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
def _CreateYumSettings(args, messages):
    """Creates a YumSettings message from input arguments."""
    if not any([args.yum_excludes, args.yum_minimal, args.yum_security, args.yum_exclusive_packages]):
        return None
    return messages.YumSettings(excludes=args.yum_excludes if args.yum_excludes else [], minimal=args.yum_minimal, security=args.yum_security, exclusivePackages=args.yum_exclusive_packages if args.yum_exclusive_packages else [])