import logging
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..rcparams import rcParams
from .base import dict_to_dataset, infer_stan_dtypes, make_attrs, requires
from .inference_data import InferenceData
def _as_set(spec):
    """Uniform representation for args which be name or list of names."""
    if spec is None:
        return []
    if isinstance(spec, str):
        return [spec]
    try:
        return set(spec.values())
    except AttributeError:
        return set(spec)