from collections.abc import Sequence
import functools
import sys
from typing import Any, NamedTuple, Optional, Protocol, TypeVar
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
class InvalidSymbolNameError(Exception):
    """Raised when trying to export symbol as an invalid or unallowed name."""