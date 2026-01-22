from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.command_lib.dataplex import parsers as dataplex_parsers
from googlecloudsdk.command_lib.util.args import labels_util
def AddEntrySourceArgs(parser: parser_arguments.ArgumentInterceptor, for_update: bool):
    """Add entry source update args.

  Args:
    parser: The arg parser to add flags to.
    for_update: If True, then indicates that arguments are intended for Update
      command. In such case for each clearable argument there will be also
      `--clear-...` flag added in a mutually exclusive group to support clearing
      the field.
  """
    entry_source = parser.add_group(help='Source system related information for an entry. If any of the entry source fields are specified, then ``--entry-source-update-time` must be specified as well.')

    def AddArgument(name: str, **kwargs):
        parser_to_add = entry_source
        if for_update:
            parser_to_add = entry_source.add_mutually_exclusive_group()
            parser_to_add.add_argument('--clear-entry-source-' + name, action='store_true', help=f'Clear the value for the {name.replace('-', '_')} field in the Entry Source.')
        parser_to_add.add_argument('--entry-source-' + name, **kwargs)
    AddArgument('resource', help='The name of the resource in the source system.', metavar='RESOURCE')
    AddArgument('system', help='The name of the source system.', metavar='SYSTEM_NAME')
    AddArgument('platform', help='The platform containing the source system.', metavar='PLATFORM_NAME')
    AddArgument('display-name', help='User friendly display name.', metavar='DISPLAY_NAME')
    AddArgument('description', help='Description of the Entry.', metavar='DESCRIPTION')
    AddArgument('create-time', help='The creation date and time of the resource in the source system.', type=dataplex_parsers.IsoDateTime, metavar='DATE_TIME')
    entry_source_labels_container = entry_source
    if for_update:
        entry_source_labels_container = entry_source.add_mutually_exclusive_group()
        clear_flag = labels_util.GetClearLabelsFlag(labels_name='entry-source-labels').AddToParser(entry_source_labels_container)
        clear_flag.help = 'Clear the labels for the Entry Source.'
    labels_util.GetCreateLabelsFlag(labels_name='entry-source-labels').AddToParser(entry_source_labels_container)
    if not for_update:
        entry_source.add_argument('--entry-source-ancestors', help='Information about individual items in the hierarchy of an Entry.', type=arg_parsers.ArgList(includes_json=True), metavar='ANCESTORS')
    entry_source.add_argument('--entry-source-update-time', help='The update date and time of the resource in the source system.', type=dataplex_parsers.IsoDateTime, required=for_update, metavar='DATE_TIME')