import logging
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from .. import utils
from ..rcparams import rcParams
from .base import CoordSpec, DimSpec, dict_to_dataset, infer_stan_dtypes, requires
from .inference_data import InferenceData
Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `output`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        