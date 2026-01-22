from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetApiResourceSpec(resource_name='api', wildcard=False):
    return concepts.ResourceSpec('apigateway.projects.locations.apis', resource_name=resource_name, apisId=ApiAttributeConfig(wildcard=wildcard), locationsId=LocationAttributeConfig(default='global'), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)