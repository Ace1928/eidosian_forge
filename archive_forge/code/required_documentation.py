from __future__ import annotations
import logging # isort:skip
from typing import Any, TypeVar
from ._sphinx import property_link, register_type_link, type_link
from .bases import (
from .singletons import Undefined
 A property accepting a value of some other type while having undefined default. 