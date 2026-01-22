from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import resources
def AdminClusterMembershipAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='admin_cluster_membership', help_text='admin cluster membership of the {resource}, in the form of projects/PROJECT/locations/global/memberships/MEMBERSHIP. ')