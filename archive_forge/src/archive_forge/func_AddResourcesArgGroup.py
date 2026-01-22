from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddResourcesArgGroup(parser):
    """Add the argument group for ResourceCommitment support in commitment."""
    resources_group = parser.add_group('Manage the commitment for particular resources.', required=True)
    resources_help = 'Resources to be included in the commitment. For details and examples of valid\nspecifications, refer to the\n[custom machine type guide](https://cloud.google.com/compute/docs/instances/creating-instance-with-custom-machine-type#specifications).\n*memory*::: The size of the memory, should include units (e.g. 3072MB or 9GB). If no units are specified, GB is assumed.\n*vcpu*::: The number of the vCPU cores.\n*local-ssd*::: The size of local SSD.\n'
    resources_group.add_argument('--resources', help=resources_help, type=arg_parsers.ArgDict(spec={'vcpu': int, 'local-ssd': int, 'memory': arg_parsers.BinarySize()}))
    accelerator_help = 'Manage the configuration of the type and number of accelerator cards to include in the commitment.\n*count*::: The number of accelerators to include.\n*type*::: The specific type (e.g. nvidia-tesla-k80 for NVIDIA Tesla K80) of the accelerator. Use `gcloud compute accelerator-types list` to learn about all available accelerator types.\n'
    resources_group.add_argument('--resources-accelerator', help=accelerator_help, type=arg_parsers.ArgDict(spec={'count': int, 'type': str}))