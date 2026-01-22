from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.command_lib.secrets import completers as secrets_completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import resources
def GetRegionalSecretResourceSpec():
    """Returns the resource spec for regional secret."""
    return concepts.ResourceSpec(resource_collection='secretmanager.projects.locations.secrets', resource_name='regional secret', plural_name='secrets', disable_auto_completers=False, secretsId=GetRegionalSecretAttributeConfig(), projectsId=GetProjectAttributeConfig(), locationsId=GetLocationResourceAttributeConfig())