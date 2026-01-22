from __future__ import annotations
import re
from typing import (
import warnings
from pandas.errors import CSSWarning
from pandas.util._exceptions import find_stack_level
def _update_initial(self, props: dict[str, str], inherited: dict[str, str]) -> dict[str, str]:
    for prop, val in inherited.items():
        if prop not in props:
            props[prop] = val
    new_props = props.copy()
    for prop, val in props.items():
        if val == 'inherit':
            val = inherited.get(prop, 'initial')
        if val in ('initial', None):
            del new_props[prop]
        else:
            new_props[prop] = val
    return new_props