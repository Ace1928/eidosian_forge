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
def AddReservationArguments(parser):
    """Add --source-reservation and --dest-reservation arguments to parser."""
    help_text = '\n{0} reservation configuration.\n*reservation*::: Name of the {0} reservation to operate on.\n*reservation-zone*:::  Zone of the {0} reservation to operate on.\n*vm-count*::: The number of VM instances that are allocated to this reservation.\nThe value of this field must be an int in the range [1, 1000].\n*machine-type*:::  The type of machine (name only) which has a fixed number of\nvCPUs and a fixed amount of memory. This also includes specifying custom machine\ntype following `custom-number_of_CPUs-amount_of_memory` pattern, e.g. `custom-32-29440`.\n*min-cpu-platform*::: Optional minimum CPU platform of the reservation to create.\n*require-specific-reservation*::: Indicates whether the reservation can be consumed by VMs with "any reservation"\ndefined. If enabled, then only VMs that target this reservation by name using\n`--reservation-affinity=specific` can consume from this reservation.\n'
    reservation_spec = {'reservation': str, 'reservation-zone': str, 'vm-count': int, 'machine-type': str, 'min-cpu-platform': str, 'require-specific-reservation': bool}
    parser.add_argument('--source-reservation', type=arg_parsers.ArgDict(spec=reservation_spec), help=help_text.format('source'), required=True)
    parser.add_argument('--dest-reservation', type=arg_parsers.ArgDict(spec=reservation_spec), help=help_text.format('destination'), required=True)
    return parser