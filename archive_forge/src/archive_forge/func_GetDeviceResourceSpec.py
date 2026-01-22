from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetDeviceResourceSpec(resource_name='device'):
    return concepts.ResourceSpec('cloudiot.projects.locations.registries.devices', resource_name=resource_name, devicesId=DeviceAttributeConfig(name=resource_name), registriesId=RegistryAttributeConfig(), locationsId=RegionAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)