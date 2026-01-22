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
def GetRegionalSecretAttributeConfig():
    """Returns the attribute config for regional secret."""
    return concepts.ResourceParameterAttributeConfig(name='secret', help_text='The secret of the {resource}.', completer=secrets_completers.SecretsCompleter)