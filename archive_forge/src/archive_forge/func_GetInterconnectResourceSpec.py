from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetInterconnectResourceSpec(resource_name='interconnect'):
    return concepts.ResourceSpec('edgenetwork.projects.locations.zones.interconnects', resource_name=resource_name, interconnectsId=InterconnectAttributeConfig(name=resource_name), zonesId=ZoneAttributeConfig('zone'), locationsId=LocationAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)