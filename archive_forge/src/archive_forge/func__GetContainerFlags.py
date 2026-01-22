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
def _GetContainerFlags(self) -> frozenset[str]:
    """_GetContainerFlags returns the configured set of per-container flags."""
    args = [self._container_arg_group]
    flag_names = []
    while args:
        arg = args.pop()
        if isinstance(arg, calliope_base.ArgumentGroup):
            args.extend(arg.arguments)
        else:
            flag_names.append(arg.name)
    return frozenset(flag_names)