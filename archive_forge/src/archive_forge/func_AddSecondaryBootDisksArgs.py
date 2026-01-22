from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddSecondaryBootDisksArgs(parser, hidden=False):
    """Adds args for secondary boot disk."""
    spec = {'mode': str, 'disk-image': str}
    parser.add_argument('--secondary-boot-disk', hidden=hidden, action='append', type=arg_parsers.ArgDict(spec=spec, required_keys=['disk-image'], max_length=len(spec)), metavar='disk-image=DISK_IMAGE,[mode=MODE]', help='      Attaches secondary boot disks to all nodes.\n\n      *disk-image*::: (Required) The full resource path to the source disk image to create the secondary boot disks from.\n\n      *mode*::: (Optional) The configuration mode for the secondary boot disks. The default value is "CONTAINER_IMAGE_CACHE."\n      ')