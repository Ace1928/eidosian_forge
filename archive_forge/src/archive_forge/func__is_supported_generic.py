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
def _is_supported_generic(type_):
    if type_ is _NO_ARGS:
        return False
    not_str = not _issubclass_safe(type_, str)
    is_enum = _issubclass_safe(type_, Enum)
    return not_str and _is_collection(type_) or _is_optional(type_) or is_union_type(type_) or is_enum