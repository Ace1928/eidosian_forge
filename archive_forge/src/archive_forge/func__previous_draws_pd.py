from collections import Counter
from typing import (
import numpy as np
import pandas as pd
from cmdstanpy.cmdstan_args import Method
from cmdstanpy.utils import build_xarray_data, flatten_chains, get_logger
from cmdstanpy.utils.stancsv import scan_generic_csv
from .mcmc import CmdStanMCMC
from .metadata import InferenceMetadata
from .mle import CmdStanMLE
from .runset import RunSet
from .vb import CmdStanVB
def _previous_draws_pd(self, vars: List[str], inc_warmup: bool) -> pd.DataFrame:
    if vars:
        sel: Union[List[str], slice] = vars
    else:
        sel = slice(None, None)
    p_fit = self.previous_fit
    if isinstance(p_fit, CmdStanMCMC):
        return p_fit.draws_pd(vars or None, inc_warmup=inc_warmup)
    elif isinstance(p_fit, CmdStanMLE):
        if inc_warmup and p_fit._save_iterations:
            return p_fit.optimized_iterations_pd[sel]
        else:
            return p_fit.optimized_params_pd[sel]
    else:
        return p_fit.variational_sample_pd[sel]