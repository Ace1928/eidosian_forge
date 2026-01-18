import keyword
import warnings
import weakref
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from itertools import islice, zip_longest
from types import BuiltinFunctionType, CodeType, FunctionType, GeneratorType, LambdaType, ModuleType
from typing import (
from typing_extensions import Annotated
from .errors import ConfigError
from .typing import (
from .version import version_info
def to_lower_camel(string: str) -> str:
    if len(string) >= 1:
        pascal_string = to_camel(string)
        return pascal_string[0].lower() + pascal_string[1:]
    return string.lower()