from __future__ import annotations
from typing import Any, Dict, List
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.dataplex import parsers as dataplex_parsers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
def _GetEntrySource(args: parser_extensions.Namespace) -> dataplex_message.GoogleCloudDataplexV1EntrySource:
    return dataplex_message.GoogleCloudDataplexV1EntrySource(resource=_GetArgValueOrNone(args, 'entry_source_resource'), system=_GetArgValueOrNone(args, 'entry_source_system'), platform=_GetArgValueOrNone(args, 'entry_source_platform'), displayName=_GetArgValueOrNone(args, 'entry_source_display_name'), description=_GetArgValueOrNone(args, 'entry_source_description'), labels=_GetEntrySourceLabels(args), ancestors=_GetEntrySourceAncestors(args), createTime=_GetArgValueOrNone(args, 'entry_source_create_time'), updateTime=_GetArgValueOrNone(args, 'entry_source_update_time'))