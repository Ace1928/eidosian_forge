import inspect, re
import types
from typing import Optional, Callable
from lark import Transformer, v_args
class AsList:
    """Abstract class

    Subclasses will be instantiated with the parse results as a single list, instead of as arguments.
    """