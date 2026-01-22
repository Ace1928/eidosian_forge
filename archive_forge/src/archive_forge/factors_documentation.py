from __future__ import annotations
import logging # isort:skip
import typing as tp
from .bases import Init, SingleParameterizedProperty
from .container import Seq, Tuple
from .either import Either
from .primitive import String
from .singletons import Intrinsic
 Represents a collection of categorical factors. 