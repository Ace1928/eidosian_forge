from __future__ import annotations
import collections
from collections.abc import Sequence
from typing import Any
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import cli
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.run import flags
def _NewContainerParser(self) -> parser_extensions.ArgumentParser:
    """_NewContainerParser creates a new parser for parsing container args."""
    parser = parser_extensions.ArgumentParser(add_help=False, prog=self._prog, calliope_command=self._calliope_command)
    ai = parser_arguments.ArgumentInterceptor(parser=parser, is_global=False, cli_generator=None, allow_positional=True)
    self._container_arg_group.AddToParser(ai)
    cli.FLAG_INTERNAL_FLAG_FILE_LINE.AddToParser(ai)
    return parser