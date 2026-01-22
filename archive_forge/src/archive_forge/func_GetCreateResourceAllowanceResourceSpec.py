from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def GetCreateResourceAllowanceResourceSpec():
    return concepts.ResourceSpec('batch.projects.locations.resourceAllowances', resource_name='resourceAllowance', api_version='v1alpha', projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=LocationAttributeConfig(), resourceAllowancesId=concepts.ResourceParameterAttributeConfig(name='resource_allowance', help_text='The resource allowance ID for the {resource}.', fallthroughs=[deps.ValueFallthrough(INVALIDID, hint='resource allowance ID is optional and will be generated if not specified')]))