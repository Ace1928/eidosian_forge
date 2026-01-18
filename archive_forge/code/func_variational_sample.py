from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from cmdstanpy.cmdstan_args import Method
from cmdstanpy.utils import scan_variational_csv
from cmdstanpy.utils.logging import get_logger
from .metadata import InferenceMetadata
from .runset import RunSet
@property
def variational_sample(self) -> np.ndarray:
    """Returns the set of approximate posterior output draws."""
    return self._variational_sample