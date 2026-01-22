from __future__ import annotations
import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple, Union
from typing_extensions import get_args, get_origin
from . import _fields, _instantiators, _resolver, _typing
from .conf import _confstruct
From an object instance, return a data structure representing the types in the object.