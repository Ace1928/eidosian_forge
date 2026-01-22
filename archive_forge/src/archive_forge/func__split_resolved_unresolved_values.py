import copy
import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy
import random
from ray.tune.search.sample import Categorical, Domain, Function, RandomState
from ray.util.annotations import DeveloperAPI, PublicAPI
def _split_resolved_unresolved_values(spec: Dict) -> Tuple[Dict[Tuple, Any], Dict[Tuple, Any]]:
    resolved_vars = {}
    unresolved_vars = {}
    for k, v in spec.items():
        resolved, v = _try_resolve(v)
        if not resolved:
            unresolved_vars[k,] = v
        elif isinstance(v, dict):
            _resolved_children, _unresolved_children = _split_resolved_unresolved_values(v)
            for path, value in _resolved_children.items():
                resolved_vars[(k,) + path] = value
            for path, value in _unresolved_children.items():
                unresolved_vars[(k,) + path] = value
        elif isinstance(v, (list, tuple)):
            for i, elem in enumerate(v):
                _resolved_children, _unresolved_children = _split_resolved_unresolved_values({i: elem})
                for path, value in _resolved_children.items():
                    resolved_vars[(k,) + path] = value
                for path, value in _unresolved_children.items():
                    unresolved_vars[(k,) + path] = value
        else:
            resolved_vars[k,] = v
    return (resolved_vars, unresolved_vars)