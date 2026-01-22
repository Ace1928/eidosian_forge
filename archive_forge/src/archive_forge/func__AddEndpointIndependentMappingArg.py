from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddEndpointIndependentMappingArg(parser):
    help_text = textwrap.dedent('  Enable endpoint-independent mapping for the NAT (as defined in RFC 5128).\n\n  If not specified, NATs have endpoint-independent mapping disabled by default.\n\n  Use `--no-enable-endpoint-independent-mapping` to disable endpoint-independent\n  mapping.\n  ')
    parser.add_argument('--enable-endpoint-independent-mapping', action='store_true', default=None, help=help_text)