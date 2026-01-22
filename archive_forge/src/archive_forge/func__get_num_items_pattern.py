import inspect
import json
import re
from typing import Callable, Optional
from jsonschema.protocols import Validator
from pydantic import create_model
from referencing import Registry, Resource
from referencing._core import Resolver
from referencing.jsonschema import DRAFT202012
def _get_num_items_pattern(min_items, max_items, whitespace_pattern):
    min_items = int(min_items or 0)
    if max_items is None:
        return f'{{{max(min_items - 1, 0)},}}'
    else:
        max_items = int(max_items)
        if max_items < 1:
            return None
        return f'{{{max(min_items - 1, 0)},{max_items - 1}}}'