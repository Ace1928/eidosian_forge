from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def GetSubmitJobResourceSpec():
    return concepts.ResourceSpec('batch.projects.locations.jobs', resource_name='job', jobsId=concepts.ResourceParameterAttributeConfig(name='job', help_text='The job ID for the {resource}.', fallthroughs=[deps.ValueFallthrough(INVALIDID, hint='job ID is optional and will be generated if not specified')]), locationsId=LocationAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)