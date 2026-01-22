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
@classmethod
def _validate_overrides(cls, filled: Config, overrides: Dict[str, Any]):
    """Validate overrides against a filled config to make sure there are
        no references to properties that don't exist and weren't used."""
    error_msg = "Invalid override: config value doesn't exist"
    errors = []
    for override_key in overrides.keys():
        if not cls._is_in_config(override_key, filled):
            errors.append({'msg': error_msg, 'loc': [override_key]})
    if errors:
        raise ConfigValidationError(config=filled, errors=errors)