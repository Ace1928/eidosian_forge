from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from . import rules_inline
from .ruler import Ruler
from .rules_inline.state_inline import StateInline
from .token import Token
from .utils import EnvType
Process input string and push inline tokens into `tokens`