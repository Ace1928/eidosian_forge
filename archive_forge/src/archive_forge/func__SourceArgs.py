from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import textwrap
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.api_lib.compute.regions import utils as region_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.disks import create
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.compute.resource_policies import flags as resource_flags
from googlecloudsdk.command_lib.compute.resource_policies import util as resource_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
import six
def _SourceArgs(parser):
    """Add mutually exclusive source args."""
    source_parent_group = parser.add_group()
    source_group = source_parent_group.add_mutually_exclusive_group()

    def AddImageHelp():
        """Returns detailed help for `--image` argument."""
        template = "        An image to apply to the disks being created. When using\n        this option, the size of the disks must be at least as large as\n        the image size. Use ``--size'' to adjust the size of the disks.\n\n        This flag is mutually exclusive with ``--source-snapshot'' and\n        ``--image-family''.\n        "
        return template
    source_group.add_argument('--image', help=AddImageHelp)
    image_utils.AddImageProjectFlag(source_parent_group)
    source_group.add_argument('--image-family', help='        The image family for the operating system that the boot disk will be\n        initialized with. Compute Engine offers multiple Linux\n        distributions, some of which are available as both regular and\n        Shielded VM images.  When a family is specified instead of an image,\n        the latest non-deprecated image associated with that family is\n        used. It is best practice to use --image-family when the latest\n        version of an image is needed.\n        ')
    image_utils.AddImageFamilyScopeFlag(source_parent_group)
    disks_flags.SOURCE_SNAPSHOT_ARG.AddArgument(source_group)
    disks_flags.SOURCE_INSTANT_SNAPSHOT_ARG.AddArgument(source_group)
    disks_flags.SOURCE_DISK_ARG.AddArgument(parser, mutex_group=source_group)
    disks_flags.ASYNC_PRIMARY_DISK_ARG.AddArgument(parser, mutex_group=source_group)
    disks_flags.AddPrimaryDiskProject(parser)
    disks_flags.AddLocationHintArg(parser)