from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import flags as forwarding_rule_flags
def AddDomainNames(parser):
    parser.add_argument('--domain-names', type=arg_parsers.ArgList(), metavar='DOMAIN_NAMES', default=None, help='      Specifies a comma separated list of DNS domain names that are used during\n      DNS integration on PSC connected endpoints.\n      ')