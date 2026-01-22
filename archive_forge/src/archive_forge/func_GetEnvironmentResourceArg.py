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
def GetEnvironmentResourceArg(api_version, help_text, positional=True, required=True):
    """Constructs and returns the Environment Resource Argument."""

    def GetEnvironmentResourceSpec():
        """Constructs and returns the Resource specification for Environment."""

        def EnvironmentAttributeConfig():
            return concepts.ResourceParameterAttributeConfig(name='environment', help_text=help_text)

        def LocationAttributeConfig():
            return concepts.ResourceParameterAttributeConfig(name='{}location'.format('' if positional else 'environment-'), help_text='Google Cloud location of this environment https://cloud.google.com/compute/docs/regions-zones/#locations.', completer=completers.LocationCompleter, fallthroughs=[deps.ArgFallthrough('--location'), deps.PropertyFallthrough(properties.VALUES.notebooks.location)])
        return concepts.ResourceSpec('notebooks.projects.locations.environments', resource_name='environment', api_version=api_version, environmentsId=EnvironmentAttributeConfig(), locationsId=LocationAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)
    return concept_parsers.ConceptParser.ForResource('{}environment'.format('' if positional else '--'), GetEnvironmentResourceSpec(), help_text, required=required)