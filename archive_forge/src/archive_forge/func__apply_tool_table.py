import logging
import os
from collections.abc import Mapping
from email.headerregistry import Address
from functools import partial, reduce
from inspect import cleandoc
from itertools import chain
from types import MappingProxyType
from typing import (
from ..errors import RemovedConfigError
from ..warnings import SetuptoolsWarning
def _apply_tool_table(dist: 'Distribution', config: dict, filename: _Path):
    tool_table = config.get('tool', {}).get('setuptools', {})
    if not tool_table:
        return
    for field, value in tool_table.items():
        norm_key = json_compatible_key(field)
        if norm_key in TOOL_TABLE_REMOVALS:
            suggestion = cleandoc(TOOL_TABLE_REMOVALS[norm_key])
            msg = f'\n            The parameter `tool.setuptools.{field}` was long deprecated\n            and has been removed from `pyproject.toml`.\n            '
            raise RemovedConfigError('\n'.join([cleandoc(msg), suggestion]))
        norm_key = TOOL_TABLE_RENAMES.get(norm_key, norm_key)
        _set_config(dist, norm_key, value)
    _copy_command_options(config, dist, filename)