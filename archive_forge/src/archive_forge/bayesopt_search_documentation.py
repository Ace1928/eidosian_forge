from collections import defaultdict
import logging
import pickle
import json
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import Domain, Float, Quantized, Uniform
from ray.tune.search import (
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils.util import is_nan_or_inf, unflatten_dict
from ray.tune.utils import flatten_dict
Restoring current optimizer state.