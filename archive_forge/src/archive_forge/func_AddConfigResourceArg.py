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
def AddConfigResourceArg(parser, api_version='v1beta', flag_anchor=False, global_fallthrough=False):
    """Create a config resource argument."""
    spec = concepts.ResourceSpec('workstations.projects.locations.workstationClusters.workstationConfigs', resource_name='config', api_version=api_version, workstationConfigsId=ConfigsAttributeConfig(global_fallthrough=global_fallthrough), workstationClustersId=ClustersAttributeConfig(cluster_fallthrough=True, global_fallthrough=global_fallthrough), locationsId=LocationsAttributeConfig(location_fallthrough=True, global_fallthrough=global_fallthrough), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)
    concept_parsers.ConceptParser.ForResource('--config' if flag_anchor else 'config', spec, 'The group of arguments defining a config', required=True).AddToParser(parser)