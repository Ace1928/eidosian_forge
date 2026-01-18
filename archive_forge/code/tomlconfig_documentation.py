from __future__ import annotations
import os
import re
from typing import Any, Callable, Iterable, TypeVar
from coverage import env
from coverage.exceptions import ConfigError
from coverage.misc import import_third_party, substitute_variables
from coverage.types import TConfigSectionOut, TConfigValueOut
Get a list of strings, substituting environment variables in the elements.