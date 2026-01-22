from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddCreateSubPrefixArgs(parser):
    """Adds flags for delegate sub prefixes create command."""
    _AddCommonSubPrefixArgs(parser, 'create')
    parser.add_argument('--range', help='IPv4 range from this public delegated prefix that should be delegated, in CIDR format. If not specified, the entire range ofthe public advertised prefix is delegated.')
    parser.add_argument('--description', help='Description of the delegated sub prefix to create.')
    parser.add_argument('--delegatee-project', help='Project to delegate the IPv4 range specified in `--range` to. If empty, the sub-range is delegated to the same/existing project.')
    parser.add_argument('--create-addresses', action='store_true', help='Specify if the sub prefix is delegated to create address resources in the delegatee project. Default is false.')