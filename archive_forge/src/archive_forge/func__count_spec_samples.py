import copy
import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy
import random
from ray.tune.search.sample import Categorical, Domain, Function, RandomState
from ray.util.annotations import DeveloperAPI, PublicAPI
def _count_spec_samples(spec: Dict, num_samples=1) -> int:
    """Count samples for a specific spec"""
    _, domain_vars, grid_vars = parse_spec_vars(spec)
    grid_count = 1
    for path, domain in grid_vars:
        grid_count *= len(domain.categories)
    return num_samples * grid_count