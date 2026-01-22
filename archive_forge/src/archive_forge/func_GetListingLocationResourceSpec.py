from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def GetListingLocationResourceSpec():
    location_attribute_config = GetZoneAttributeConfig()
    location_attribute_config.fallthroughs.insert(0, deps.Fallthrough(lambda: '-', hint='uses all locations by default.'))
    return concepts.ResourceSpec('file.projects.locations', 'zone', projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=location_attribute_config)