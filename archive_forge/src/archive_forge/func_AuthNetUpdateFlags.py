from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.active_directory import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AuthNetUpdateFlags():
    """Defines flags for updating authorized networks."""
    auth_net_group = base.ArgumentGroup(mutex=True)
    auth_net_group.AddArgument(DomainAddAuthorizedNetworksFlag())
    auth_net_group.AddArgument(DomainRemoveAuthorizedNetworksFlag())
    return auth_net_group