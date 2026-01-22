import itertools
import six
import numpy as np
from patsy import PatsyError
from patsy.categorical import (guess_categorical,
from patsy.util import (atleast_2d_column_default,
from patsy.design_info import (DesignMatrix, DesignInfo,
from patsy.redundancy import pick_contrasts_for_term
from patsy.eval import EvalEnvironment
from patsy.contrasts import code_contrast_matrix, Treatment
from patsy.compat import OrderedDict
from patsy.missing import NAAction
def _factors_memorize(factors, data_iter_maker, eval_env):
    factor_states = {}
    passes_needed = {}
    for factor in factors:
        state = {}
        which_pass = factor.memorize_passes_needed(state, eval_env)
        factor_states[factor] = state
        passes_needed[factor] = which_pass
    memorize_needed = set()
    for factor, passes in six.iteritems(passes_needed):
        if passes > 0:
            memorize_needed.add(factor)
    which_pass = 0
    while memorize_needed:
        for data in data_iter_maker():
            for factor in memorize_needed:
                state = factor_states[factor]
                factor.memorize_chunk(state, which_pass, data)
        for factor in list(memorize_needed):
            factor.memorize_finish(factor_states[factor], which_pass)
            if which_pass == passes_needed[factor] - 1:
                memorize_needed.remove(factor)
        which_pass += 1
    return factor_states