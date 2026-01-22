from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import update_args
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import resources
def _GetResourceArgGenerator(arg_data, resource_collection, shared_resource_args):
    """Gets a function to generate a resource arg."""
    ignored_attributes = _GetAllSharedAttributes(arg_data, shared_resource_args)

    def ArgGen(name, group_help):
        group_help += '\n\n'
        if arg_data.group_help:
            group_help += arg_data.group_help
        return arg_data.GenerateResourceArg(resource_collection, name, ignored_attributes, group_help=group_help)
    return ArgGen