from __future__ import annotations
import argparse
import contextlib
import dataclasses
import difflib
import itertools
import re as _re
import shlex
import shutil
import sys
from gettext import gettext as _
from typing import Any, Dict, Generator, Iterable, List, NoReturn, Optional, Set, Tuple
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from typing_extensions import override
from . import _arguments, _strings, conf
from ._parsers import ParserSpecification
@dataclasses.dataclass
class TyroTheme:
    border: Style = Style()
    description: Style = Style()
    invocation: Style = Style()
    metavar: Style = Style()
    metavar_fixed: Style = Style()
    helptext: Style = Style()
    helptext_required: Style = Style()
    helptext_default: Style = Style()

    def as_rich_theme(self) -> Theme:
        return Theme(vars(self))