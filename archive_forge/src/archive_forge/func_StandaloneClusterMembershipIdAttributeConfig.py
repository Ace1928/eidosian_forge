from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def StandaloneClusterMembershipIdAttributeConfig():
    """Gets standalone cluster membership ID resource attribute."""
    return concepts.ResourceParameterAttributeConfig(name='membership', help_text=' membership of the {resource}, in the form of projects/PROJECT/locations/LOCATION/memberships/MEMBERSHIP. ')