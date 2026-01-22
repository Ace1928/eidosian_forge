from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.util import scaled_integer
import six
def AddAcceleratorArgs(parser):
    """Adds Accelerator-related args."""
    parser.add_argument('--accelerator', type=arg_parsers.ArgDict(spec={'type': str, 'count': int}), help="      Attaches accelerators (e.g. GPUs) to the node template.\n\n      *type*::: The specific type (e.g. nvidia-tesla-k80 for nVidia Tesla K80)\n      of accelerator to attach to the node template. Use 'gcloud compute\n      accelerator-types list' to learn about all available accelerator types.\n\n      *count*::: Number of accelerators to attach to each\n      node template. The default value is 1.\n      ")