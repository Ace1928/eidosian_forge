from collections import Counter, OrderedDict
from functools import partial
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pandas.api.types
from ray.air.util.data_batch_conversion import BatchFormat
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor, PreprocessorNotFittedException
from ray.util.annotations import PublicAPI
Inverse transform the given dataset.

        Args:
            ds: Input Dataset that has been fitted and/or transformed.

        Returns:
            ray.data.Dataset: The inverse transformed Dataset.

        Raises:
            PreprocessorNotFittedException: if ``fit`` is not called yet.
        