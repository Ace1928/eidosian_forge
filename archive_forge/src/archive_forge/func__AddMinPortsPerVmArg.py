from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddMinPortsPerVmArg(parser, for_create=False):
    """Adds an argument to specify the minimum number of ports per VM for NAT."""
    help_text = textwrap.dedent('  Minimum ports to be allocated to a VM.\n\n  If Dynamic Port Allocation is disabled, this defaults to 64.\n\n  If Dynamic Port Allocation is enabled, this defaults to 32 and must be set\n  to a power of 2 that is at least 32 and lower than maxPortsPerVm.\n  ')
    _AddClearableArgument(parser, for_create, 'min-ports-per-vm', arg_parsers.BoundedInt(lower_bound=2), help_text, 'Clear minimum ports to be allocated to a VM')