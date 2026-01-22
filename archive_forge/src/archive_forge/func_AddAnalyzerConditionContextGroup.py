from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddAnalyzerConditionContextGroup(parser):
    """Adds a group of options."""
    condition_context_group = parser.add_group(mutex=False, required=False, help='The hypothetical context to evaluate IAM conditions.')
    AddAnalyzerAccessTimeArgs(condition_context_group)