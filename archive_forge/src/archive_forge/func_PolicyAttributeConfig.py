from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def PolicyAttributeConfig(api_version):
    return concepts.ResourceParameterAttributeConfig(name='policy', completer=PolicyCompleter(api_version=api_version), help_text='The Cloud DNS policy name {resource}.')