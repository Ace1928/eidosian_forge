from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddWorkstationResourceArg(parser, api_version='v1beta'):
    """Create a workstation resource argument."""
    spec = concepts.ResourceSpec('workstations.projects.locations.workstationClusters.workstationConfigs.workstations', resource_name='workstation', api_version=api_version, workstationsId=WorkstationsAttributeConfig(), workstationConfigsId=ConfigsAttributeConfig(config_fallthrough=True), workstationClustersId=ClustersAttributeConfig(cluster_fallthrough=True), locationsId=LocationsAttributeConfig(location_fallthrough=True), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)
    concept_parsers.ConceptParser.ForResource('workstation', spec, 'The group of arguments defining a workstation', required=True).AddToParser(parser)