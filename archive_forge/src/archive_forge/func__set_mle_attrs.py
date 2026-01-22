from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from cmdstanpy.cmdstan_args import Method, OptimizeArgs
from cmdstanpy.utils import get_logger, scan_optimize_csv
from .metadata import InferenceMetadata
from .runset import RunSet
def _set_mle_attrs(self, sample_csv_0: str) -> None:
    meta = scan_optimize_csv(sample_csv_0, self._save_iterations)
    self._metadata = InferenceMetadata(meta)
    self._column_names: Tuple[str, ...] = meta['column_names']
    self._mle: np.ndarray = meta['mle']
    if self._save_iterations:
        self._all_iters: np.ndarray = meta['all_iters']