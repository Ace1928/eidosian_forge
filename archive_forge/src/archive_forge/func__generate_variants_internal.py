import copy
import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy
import random
from ray.tune.search.sample import Categorical, Domain, Function, RandomState
from ray.util.annotations import DeveloperAPI, PublicAPI
def _generate_variants_internal(spec: Dict, constant_grid_search: bool=False, random_state: 'RandomState'=None) -> Tuple[Dict, Dict]:
    spec = copy.deepcopy(spec)
    _, domain_vars, grid_vars = parse_spec_vars(spec)
    if not domain_vars and (not grid_vars):
        yield ({}, spec)
        return
    to_resolve = domain_vars
    all_resolved = True
    if constant_grid_search:
        all_resolved, resolved_vars = _resolve_domain_vars(spec, domain_vars, allow_fail=True, random_state=random_state)
        if not all_resolved:
            to_resolve = [(r, d) for r, d in to_resolve if r not in resolved_vars]
    grid_search = _grid_search_generator(spec, grid_vars)
    for resolved_spec in grid_search:
        if not constant_grid_search or not all_resolved:
            _, resolved_vars = _resolve_domain_vars(resolved_spec, to_resolve, random_state=random_state)
        for resolved, spec in _generate_variants_internal(resolved_spec, constant_grid_search=constant_grid_search, random_state=random_state):
            for path, value in grid_vars:
                resolved_vars[path] = _get_value(spec, path)
            for k, v in resolved.items():
                if k in resolved_vars and v != resolved_vars[k] and _is_resolved(resolved_vars[k]):
                    raise ValueError('The variable `{}` could not be unambiguously resolved to a single value. Consider simplifying your configuration.'.format(k))
                resolved_vars[k] = v
            yield (resolved_vars, spec)