from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from cmdstanpy.cmdstan_args import Method, OptimizeArgs
from cmdstanpy.utils import get_logger, scan_optimize_csv
from .metadata import InferenceMetadata
from .runset import RunSet

        Move output CSV files to specified directory.  If files were
        written to the temporary session directory, clean filename.
        E.g., save 'bernoulli-201912081451-1-5nm6as7u.csv' as
        'bernoulli-201912081451-1.csv'.

        :param dir: directory path

        See Also
        --------
        stanfit.RunSet.save_csvfiles
        cmdstanpy.from_csv
        