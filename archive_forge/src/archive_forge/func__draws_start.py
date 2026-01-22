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
def _draws_start(self, inc_warmup: bool) -> Tuple[int, int]:
    draw1 = 0
    p_fit = self.previous_fit
    if isinstance(p_fit, CmdStanMCMC):
        num_draws = p_fit.num_draws_sampling
        if p_fit._save_warmup:
            if inc_warmup:
                num_draws += p_fit.num_draws_warmup
            else:
                draw1 = p_fit.num_draws_warmup
    elif isinstance(p_fit, CmdStanMLE):
        num_draws = 1
        if p_fit._save_iterations:
            opt_iters = len(p_fit.optimized_iterations_np)
            if inc_warmup:
                num_draws = opt_iters
            else:
                draw1 = opt_iters - 1
    else:
        draw1 = 1
        num_draws = p_fit.variational_sample.shape[0]
        if inc_warmup:
            num_draws += 1
    return (draw1, num_draws)