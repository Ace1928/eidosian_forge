import configparser
import contextlib
import locale
import logging
import pathlib
import re
import sys
from itertools import chain, groupby, repeat
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Union
from pip._vendor.requests.models import Request, Response
from pip._vendor.rich.console import Console, ConsoleOptions, RenderResult
from pip._vendor.rich.markup import escape
from pip._vendor.rich.text import Text
class MissingPyProjectBuildRequires(DiagnosticPipError):
    """Raised when pyproject.toml has `build-system`, but no `build-system.requires`."""
    reference = 'missing-pyproject-build-system-requires'

    def __init__(self, *, package: str) -> None:
        super().__init__(message=f'Can not process {escape(package)}', context=Text('This package has an invalid pyproject.toml file.\nThe [build-system] table is missing the mandatory `requires` key.'), note_stmt='This is an issue with the package mentioned above, not pip.', hint_stmt=Text('See PEP 518 for the detailed specification.'))