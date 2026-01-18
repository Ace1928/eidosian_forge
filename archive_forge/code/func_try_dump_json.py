import copy
import inspect
import io
import re
import warnings
from configparser import (
from dataclasses import dataclass
from pathlib import Path
from types import GeneratorType
from typing import (
import srsly
from .util import SimpleFrozenDict, SimpleFrozenList  # noqa: F401
def try_dump_json(value: Any, data: Union[Dict[str, dict], Config, str]='') -> str:
    """Dump a config value as JSON and output user-friendly error if it fails."""
    if isinstance(value, str) and VARIABLE_RE.search(value):
        return value
    if isinstance(value, str) and value.replace('.', '', 1).isdigit():
        value = f'"{value}"'
    try:
        value = srsly.json_dumps(value)
        value = re.sub('\\$([^{])', '$$\x01', value)
        value = re.sub('\\$$', '$$', value)
        return value
    except Exception as e:
        err_msg = f"Couldn't serialize config value of type {type(value)}: {e}. Make sure all values in your config are JSON-serializable. If you want to include Python objects, use a registered function that returns the object instead."
        raise ConfigValidationError(config=data, desc=err_msg) from e