import logging
from typing import Callable, Optional
import numpy as np
from .. import utils
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import InferenceData
Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `trace`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        