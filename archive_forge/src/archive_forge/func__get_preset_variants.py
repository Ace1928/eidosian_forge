import copy
import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy
import random
from ray.tune.search.sample import Categorical, Domain, Function, RandomState
from ray.util.annotations import DeveloperAPI, PublicAPI
def _get_preset_variants(spec: Dict, config: Dict, constant_grid_search: bool=False, random_state: 'RandomState'=None):
    """Get variants according to a spec, initialized with a config.

    Variables from the spec are overwritten by the variables in the config.
    Thus, we may end up with less sampled parameters.

    This function also checks if values used to overwrite search space
    parameters are valid, and logs a warning if not.
    """
    spec = copy.deepcopy(spec)
    resolved, _, _ = parse_spec_vars(config)
    for path, val in resolved:
        try:
            domain = _get_value(spec['config'], path)
            if isinstance(domain, dict):
                if 'grid_search' in domain:
                    domain = Categorical(domain['grid_search'])
                else:
                    domain = None
        except IndexError as exc:
            raise ValueError(f'Pre-set config key `{'/'.join(path)}` does not correspond to a valid key in the search space definition. Please add this path to the `param_space` variable passed to `tune.Tuner()`.') from exc
        if domain:
            if isinstance(domain, Domain):
                if not domain.is_valid(val):
                    logger.warning(f'Pre-set value `{val}` is not within valid values of parameter `{'/'.join(path)}`: {domain.domain_str}')
            elif domain != val:
                logger.warning(f'Pre-set value `{val}` is not equal to the value of parameter `{'/'.join(path)}`: {domain}')
        assign_value(spec['config'], path, val)
    return _generate_variants_internal(spec, constant_grid_search=constant_grid_search, random_state=random_state)