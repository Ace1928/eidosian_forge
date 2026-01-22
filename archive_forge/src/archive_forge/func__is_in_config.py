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
def _is_in_config(cls, prop: str, config: Union[Dict[str, Any], Config]):
    """Check whether a nested config property like "section.subsection.key"
        is in a given config."""
    tree = prop.split('.')
    obj = dict(config)
    while tree:
        key = tree.pop(0)
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
        else:
            return False
    return True