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
def _set_overrides(self, config: 'ConfigParser', overrides: Dict[str, Any]) -> None:
    """Set overrides in the ConfigParser before config is interpreted."""
    err_title = 'Error parsing config overrides'
    for key, value in overrides.items():
        err_msg = 'not a section value that can be overridden'
        err = [{'loc': key.split('.'), 'msg': err_msg}]
        if '.' not in key:
            raise ConfigValidationError(errors=err, title=err_title)
        section, option = key.rsplit('.', 1)
        if section not in config:
            raise ConfigValidationError(errors=err, title=err_title)
        config.set(section, option, try_dump_json(value, overrides))