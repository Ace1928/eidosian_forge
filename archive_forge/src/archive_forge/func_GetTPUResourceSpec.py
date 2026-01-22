from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
def GetTPUResourceSpec(resource_name='TPU'):
    return concepts.ResourceSpec('tpu.projects.locations.nodes', resource_name=resource_name, locationsId=ZoneAttributeConfig(), nodesId=TPUAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)