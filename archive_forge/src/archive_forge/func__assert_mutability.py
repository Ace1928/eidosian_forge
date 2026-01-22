from __future__ import annotations
import copy
import json
from importlib import import_module
from pprint import pformat
from types import ModuleType
from typing import (
from scrapy.settings import default_settings
def _assert_mutability(self) -> None:
    if self.frozen:
        raise TypeError('Trying to modify an immutable Settings object')