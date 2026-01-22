from __future__ import annotations
from typing import Any, Dict, List
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.dataplex import parsers as dataplex_parsers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
def _GetEntrySourceAncestors(args: parser_extensions.Namespace) -> List[Any]:
    """Parse EntrySource ancestors from the command arguments if defined."""
    if not args.IsKnownAndSpecified('entry_source_ancestors'):
        return []
    return dataplex_parsers.ParseEntrySourceAncestors(args.entry_source_ancestors)