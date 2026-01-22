import copy
import json
import sys
import warnings
from collections import defaultdict, namedtuple
from dataclasses import (MISSING,
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import (Any, Collection, Mapping, Union, get_type_hints,
from uuid import UUID
from typing_inspect import is_union_type  # type: ignore
from dataclasses_json import cfg
from dataclasses_json.utils import (_get_type_cons, _get_type_origin,
def _decode_item(type_arg, x):
    if is_dataclass(type_arg) or is_dataclass(xs):
        return _decode_dataclass(type_arg, x, infer_missing)
    if _is_supported_generic(type_arg):
        return _decode_generic(type_arg, x, infer_missing)
    return x