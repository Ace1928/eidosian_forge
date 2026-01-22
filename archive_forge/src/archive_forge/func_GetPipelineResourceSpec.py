from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetPipelineResourceSpec():
    """Constructs and returns the Resource specification for Pipeline."""

    def PipelineAttributeConfig():
        return concepts.ResourceParameterAttributeConfig(name=arg_name, help_text=help_text)
    return concepts.ResourceSpec('datapipelines.projects.locations.pipelines', resource_name='pipeline', pipelinesId=PipelineAttributeConfig(), locationsId=RegionAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)