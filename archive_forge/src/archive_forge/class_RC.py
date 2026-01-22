from __future__ import annotations
import json
import pathlib
import platform
import click
from jsonschema import ValidationError
from rich.console import Console
from rich.json import JSON
from rich.markup import escape
from rich.padding import Padding
from rich.style import Style
from jupyter_events.schema import EventSchema, EventSchemaFileAbsent, EventSchemaLoadingError
class RC:
    """Return code enum."""
    OK = 0
    INVALID = 1
    UNPARSABLE = 2
    NOT_FOUND = 3