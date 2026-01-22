from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddEditionFlag(parser):
    """Adds a --edition flag to the given parser."""
    edition_flag = base.ChoiceArgument('--edition', required=False, choices={'enterprise': 'Enterprise is the standard option for smaller instances.', 'enterprise-plus': 'Enterprise plus option recommended for cpu-intensive workloads. Offers access to premium features and capabilities.'}, default=None, help_str='Specifies edition.')
    edition_flag.AddToParser(parser)