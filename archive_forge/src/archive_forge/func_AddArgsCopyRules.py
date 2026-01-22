from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddArgsCopyRules(parser):
    """Adds the argument for security policy copy rules."""
    parser.add_argument('--source-security-policy', required=True, help='The URL of the source security policy to copy the rules from.')
    parser.add_argument('--organization', help='Organization in which the organization security policy to copy the rules to. Must be set if security-policy is display name.')