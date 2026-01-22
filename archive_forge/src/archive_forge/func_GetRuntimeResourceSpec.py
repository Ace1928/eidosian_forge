from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.compute.networks import flags as compute_network_flags
from googlecloudsdk.command_lib.compute.networks.subnets import flags as compute_subnet_flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.notebooks import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetRuntimeResourceSpec():
    """Constructs and returns the Resource specification for Runtime."""

    def RuntimeAttributeConfig():
        return concepts.ResourceParameterAttributeConfig(name='runtime', help_text=help_text)

    def LocationAttributeConfig():
        return concepts.ResourceParameterAttributeConfig(name='location', help_text='Google Cloud location of this runtime https://cloud.google.com/compute/docs/regions-zones/#locations.', fallthroughs=[deps.PropertyFallthrough(properties.VALUES.notebooks.location)])
    return concepts.ResourceSpec('notebooks.projects.locations.runtimes', resource_name='runtime', api_version=api_version, projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=LocationAttributeConfig(), runtimesId=RuntimeAttributeConfig(), disable_auto_completers=False)