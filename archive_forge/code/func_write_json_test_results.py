from __future__ import annotations
import collections.abc as c
import contextlib
import json
import os
import re
import shlex
import sys
import tempfile
import textwrap
import typing as t
from .constants import (
from .encoding import (
from .util import (
from .io import (
from .data import (
from .provider.layout import (
from .host_configs import (
def write_json_test_results(category: ResultType, name: str, content: t.Union[list[t.Any], dict[str, t.Any]], formatted: bool=True, encoder: t.Optional[t.Type[json.JSONEncoder]]=None) -> None:
    """Write the given json content to the specified test results path, creating directories as needed."""
    path = os.path.join(category.path, name)
    write_json_file(path, content, create_directories=True, formatted=formatted, encoder=encoder)