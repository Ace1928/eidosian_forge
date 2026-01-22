import logging
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..rcparams import rcParams
from .base import dict_to_dataset, infer_stan_dtypes, make_attrs, requires
from .inference_data import InferenceData
def _filter_columns(columns, spec):
    """Parse variable name from column label, removing element index, if any."""
    return [col for col in columns if col.split('[')[0].split('.')[0] in spec]