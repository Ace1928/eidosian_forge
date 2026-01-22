import contextlib
import datetime
import ipaddress
import json
import math
from fractions import Fraction
from typing import Callable, Dict, Type, Union, cast, overload
import hypothesis.strategies as st
import pydantic
import pydantic.color
import pydantic.types
from pydantic.utils import lenient_issubclass
def _registered(typ: Union[Type[pydantic.types.T], pydantic.types.ConstrainedNumberMeta]) -> Union[Type[pydantic.types.T], pydantic.types.ConstrainedNumberMeta]:
    pydantic.types._DEFINED_TYPES.add(typ)
    for supertype, resolver in RESOLVERS.items():
        if issubclass(typ, supertype):
            st.register_type_strategy(typ, resolver(typ))
            return typ
    raise NotImplementedError(f'Unknown type {typ!r} has no resolver to register')