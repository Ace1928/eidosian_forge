from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import ipaddr
def GetPrivateForwardingTargetsArg():
    return base.Argument('--private-forwarding-targets', type=arg_parsers.ArgList(), metavar='IP_ADDRESSES', help='List of IPv4 addresses of target name servers that the zone will forward queries to. Ignored for `public` visibility. All addresses specified for this parameter will be reached through the VPC.')