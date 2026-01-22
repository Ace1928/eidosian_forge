from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetTypeFlag():
    """Anthos auth token type flag, specifies the type of token to be created."""
    return base.ChoiceArgument('--type', required=True, choices=['aws', 'oidc'], help_str='Type of token to be created.')