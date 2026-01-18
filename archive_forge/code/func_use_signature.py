from collections.abc import Mapping, MutableMapping
from copy import deepcopy
import json
import itertools
import re
import sys
import traceback
import warnings
from typing import (
from types import ModuleType
import jsonschema
import pandas as pd
import numpy as np
from pandas.api.types import infer_dtype
from altair.utils.schemapi import SchemaBase
from altair.utils._dfi_types import Column, DtypeKind, DataFrame as DfiDataFrame
from typing import Literal, Protocol, TYPE_CHECKING, runtime_checkable
def use_signature(Obj: Callable[P, Any]):
    """Apply call signature and documentation of Obj to the decorated method"""

    def decorate(f: Callable[..., V]) -> Callable[P, V]:
        f.__wrapped__ = Obj.__init__
        f._uses_signature = Obj
        if Obj.__doc__:
            doclines = Obj.__doc__.splitlines()
            doclines[0] = f'Refer to :class:`{Obj.__name__}`'
            if f.__doc__:
                doc = f.__doc__ + '\n'.join(doclines[1:])
            else:
                doc = '\n'.join(doclines)
            try:
                f.__doc__ = doc
            except AttributeError:
                pass
        return f
    return decorate