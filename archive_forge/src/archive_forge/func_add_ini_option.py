from __future__ import annotations
import sys
import warnings
from typing import Any, Literal
from pytest import Config, Parser
from typeguard._config import CollectionCheckStrategy, ForwardRefPolicy, global_config
from typeguard._exceptions import InstrumentationWarning
from typeguard._importhook import install_import_hook
from typeguard._utils import qualified_name, resolve_reference
def add_ini_option(opt_type: Literal['string', 'paths', 'pathlist', 'args', 'linelist', 'bool'] | None) -> None:
    parser.addini(group.options[-1].names()[0][2:], group.options[-1].attrs()['help'], opt_type)