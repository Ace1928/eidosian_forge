from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetTemplateTypeArg(required=False):
    choices = {'flex': 'Specifies a Flex template.', 'classic': 'Specifies a Classic template'}
    return base.ChoiceArgument('--template-type', choices=choices, required=required, default='FLEX', help_str="Type of the template. Defaults to flex template. One of 'FLEX' or 'CLASSIC'")