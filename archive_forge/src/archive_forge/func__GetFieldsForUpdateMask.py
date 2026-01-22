from __future__ import annotations
from typing import Any, Dict, List
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.dataplex import parsers as dataplex_parsers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
def _GetFieldsForUpdateMask(args: parser_extensions.Namespace) -> List[str]:
    """Create a sorted list of fields to be used in update_mask for Entry based on arguments provided to the command."""
    arg_name_to_field = {'--fully-qualified-name': 'fully_qualified_name', '--update-aspects': 'aspects', '--remove-aspects': 'aspects', '--aspects': 'aspects', '--keys': 'aspects', '--entry-source-resource': 'entry_source.resource', '--entry-source-system': 'entry_source.system', '--entry-source-platform': 'entry_source.platform', '--entry-source-display-name': 'entry_source.display_name', '--entry-source-description': 'entry_source.description', '--entry-source-labels': 'entry_source.labels', '--entry-source-create-time': 'entry_source.create_time', '--entry-source-update-time': 'entry_source.update_time'}
    args_cleaned = set(map(lambda arg: arg.replace('--clear-', '--'), args.GetSpecifiedArgNames()))
    updatable_args = args_cleaned.intersection(arg_name_to_field)
    return sorted(set(map(lambda arg_name: arg_name_to_field[arg_name], updatable_args)))