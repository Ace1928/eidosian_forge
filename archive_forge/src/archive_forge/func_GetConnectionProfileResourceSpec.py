from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetConnectionProfileResourceSpec(resource_name='connection_profile'):
    return concepts.ResourceSpec('datastream.projects.locations.connectionProfiles', resource_name=resource_name, connectionProfilesId=ConnectionProfileAttributeConfig(name=resource_name), locationsId=LocationAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=True)