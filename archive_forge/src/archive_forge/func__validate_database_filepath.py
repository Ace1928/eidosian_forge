import os
import pathlib
import uuid
from typing import Any, Dict, List, NewType, Optional, Union, cast
from dataclasses import dataclass, fields
from jupyter_core.utils import ensure_async
from tornado import web
from traitlets import Instance, TraitError, Unicode, validate
from traitlets.config.configurable import LoggingConfigurable
from jupyter_server.traittypes import InstanceFromClasses
@validate('database_filepath')
def _validate_database_filepath(self, proposal):
    """Validate a database file path."""
    value = proposal['value']
    if value == ':memory:':
        return value
    path = pathlib.Path(value)
    if path.exists():
        if path.is_dir():
            msg = '`database_filepath` expected a file path, but the given path is a directory.'
            raise TraitError(msg)
        with open(value, 'rb') as f:
            header = f.read(100)
        if not header.startswith(b'SQLite format 3') and header != b'':
            msg = 'The given file is not an SQLite database file.'
            raise TraitError(msg)
    return value