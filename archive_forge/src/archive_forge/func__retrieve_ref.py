from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set
def _retrieve_ref(path: str, schema: dict) -> dict:
    components = path.split('/')
    if components[0] != '#':
        raise ValueError('ref paths are expected to be URI fragments, meaning they should start with #.')
    out = schema
    for component in components[1:]:
        if component.isdigit():
            out = out[int(component)]
        else:
            out = out[component]
    return deepcopy(out)