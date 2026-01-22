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
def _support_extended_types(field_type, field_value):
    if _issubclass_safe(field_type, datetime):
        if isinstance(field_value, datetime):
            res = field_value
        else:
            tz = datetime.now(timezone.utc).astimezone().tzinfo
            res = datetime.fromtimestamp(field_value, tz=tz)
    elif _issubclass_safe(field_type, Decimal):
        res = field_value if isinstance(field_value, Decimal) else Decimal(field_value)
    elif _issubclass_safe(field_type, UUID):
        res = field_value if isinstance(field_value, UUID) else UUID(field_value)
    elif _issubclass_safe(field_type, (int, float, str, bool)):
        res = field_value if isinstance(field_value, field_type) else field_type(field_value)
    else:
        res = field_value
    return res