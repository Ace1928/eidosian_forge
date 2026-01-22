import itertools
import sys
import time
from numbers import Integral, Real
from warnings import warn
import numpy as np
from joblib import effective_n_jobs
from scipy import linalg
from ..base import (
from ..linear_model import Lars, Lasso, LassoLars, orthogonal_mp_gram
from ..utils import check_array, check_random_state, gen_batches, gen_even_slices
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.extmath import randomized_svd, row_norms, svd_flip
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted
def _dict_learning(X, n_components, *, alpha, max_iter, tol, method, n_jobs, dict_init, code_init, callback, verbose, random_state, return_n_iter, positive_dict, positive_code, method_max_iter):
    """Main dictionary learning algorithm"""
    t0 = time.time()
    if code_init is not None and dict_init is not None:
        code = np.array(code_init, order='F')
        dictionary = dict_init
    else:
        code, S, dictionary = linalg.svd(X, full_matrices=False)
        code, dictionary = svd_flip(code, dictionary)
        dictionary = S[:, np.newaxis] * dictionary
    r = len(dictionary)
    if n_components <= r:
        code = code[:, :n_components]
        dictionary = dictionary[:n_components, :]
    else:
        code = np.c_[code, np.zeros((len(code), n_components - r))]
        dictionary = np.r_[dictionary, np.zeros((n_components - r, dictionary.shape[1]))]
    dictionary = np.asfortranarray(dictionary)
    errors = []
    current_cost = np.nan
    if verbose == 1:
        print('[dict_learning]', end=' ')
    ii = -1
    for ii in range(max_iter):
        dt = time.time() - t0
        if verbose == 1:
            sys.stdout.write('.')
            sys.stdout.flush()
        elif verbose:
            print('Iteration % 3i (elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)' % (ii, dt, dt / 60, current_cost))
        code = sparse_encode(X, dictionary, algorithm=method, alpha=alpha, init=code, n_jobs=n_jobs, positive=positive_code, max_iter=method_max_iter, verbose=verbose)
        _update_dict(dictionary, X, code, verbose=verbose, random_state=random_state, positive=positive_dict)
        current_cost = 0.5 * np.sum((X - code @ dictionary) ** 2) + alpha * np.sum(np.abs(code))
        errors.append(current_cost)
        if ii > 0:
            dE = errors[-2] - errors[-1]
            if dE < tol * errors[-1]:
                if verbose == 1:
                    print('')
                elif verbose:
                    print('--- Convergence reached after %d iterations' % ii)
                break
        if ii % 5 == 0 and callback is not None:
            callback(locals())
    if return_n_iter:
        return (code, dictionary, errors, ii + 1)
    else:
        return (code, dictionary, errors)