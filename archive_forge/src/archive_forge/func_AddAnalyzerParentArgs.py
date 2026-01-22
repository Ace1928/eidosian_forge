from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddAnalyzerParentArgs(parser):
    """Adds analysis parent(aka scope) argument."""
    parent_group = parser.add_mutually_exclusive_group(required=True)
    AddOrganizationArgs(parent_group, 'Organization ID on which to perform the analysis. Only policies defined at or below this organization  will be targeted in the analysis.')
    AddFolderArgs(parent_group, 'Folder ID on which to perform the analysis. Only policies defined at or below this folder will be  targeted in the analysis.')
    AddProjectArgs(parent_group, 'Project ID or number on which to perform the analysis. Only policies defined at or below this project will be  targeted in the analysis.')