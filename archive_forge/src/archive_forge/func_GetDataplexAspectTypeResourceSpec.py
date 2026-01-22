from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetDataplexAspectTypeResourceSpec():
    """Gets Aspect Type resource spec."""
    return concepts.ResourceSpec('dataplex.projects.locations.aspectTypes', resource_name='aspect_type', projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=LocationAttributeConfig(), entryGroupsId=AspectTypeAttributeConfig())