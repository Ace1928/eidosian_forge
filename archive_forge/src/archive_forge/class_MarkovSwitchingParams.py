import warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, bic, hqic
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tools.tools import Bunch, pinv_extended
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.regime_switching._hamilton_filter import (
from statsmodels.tsa.regime_switching._kim_smoother import (
from statsmodels.tsa.statespace.tools import (
class MarkovSwitchingParams:
    """
    Class to hold parameters in Markov switching models

    Parameters
    ----------
    k_regimes : int
        The number of regimes between which parameters may switch.

    Notes
    -----

    The purpose is to allow selecting parameter indexes / slices based on
    parameter type, regime number, or both.

    Parameters are lexicographically ordered in the following way:

    1. Named type string (e.g. "autoregressive")
    2. Number (e.g. the first autoregressive parameter, then the second)
    3. Regime (if applicable)

    Parameter blocks are set using dictionary setter notation where the key
    is the named type string and the value is a list of boolean values
    indicating whether a given parameter is switching or not.

    For example, consider the following code:

        parameters = MarkovSwitchingParams(k_regimes=2)
        parameters['regime_transition'] = [1,1]
        parameters['exog'] = [0, 1]

    This implies the model has 7 parameters: 4 "regime_transition"-related
    parameters (2 parameters that each switch according to regimes) and 3
    "exog"-related parameters (1 parameter that does not switch, and one 1 that
    does).

    The order of parameters is then:

    1. The first "regime_transition" parameter, regime 0
    2. The first "regime_transition" parameter, regime 1
    3. The second "regime_transition" parameter, regime 1
    4. The second "regime_transition" parameter, regime 1
    5. The first "exog" parameter
    6. The second "exog" parameter, regime 0
    7. The second "exog" parameter, regime 1

    Retrieving indexes / slices is done through dictionary getter notation.
    There are three options for the dictionary key:

    - Regime number (zero-indexed)
    - Named type string (e.g. "autoregressive")
    - Regime number and named type string

    In the above example, consider the following getters:

    >>> parameters[0]
    array([0, 2, 4, 6])
    >>> parameters[1]
    array([1, 3, 5, 6])
    >>> parameters['exog']
    slice(4, 7, None)
    >>> parameters[0, 'exog']
    [4, 6]
    >>> parameters[1, 'exog']
    [4, 7]

    Notice that in the last two examples, both lists of indexes include 4.
    That's because that is the index of the the non-switching first "exog"
    parameter, which should be selected regardless of the regime.

    In addition to the getter, the `k_parameters` attribute is an dict
    with the named type strings as the keys. It can be used to get the total
    number of parameters of each type:

    >>> parameters.k_parameters['regime_transition']
    4
    >>> parameters.k_parameters['exog']
    3
    """

    def __init__(self, k_regimes):
        self.k_regimes = k_regimes
        self.k_params = 0
        self.k_parameters = {}
        self.switching = {}
        self.slices_purpose = {}
        self.relative_index_regime_purpose = [{} for i in range(self.k_regimes)]
        self.index_regime_purpose = [{} for i in range(self.k_regimes)]
        self.index_regime = [[] for i in range(self.k_regimes)]

    def __getitem__(self, key):
        _type = type(key)
        if _type is str:
            return self.slices_purpose[key]
        elif _type is int:
            return self.index_regime[key]
        elif _type is tuple:
            if not len(key) == 2:
                raise IndexError('Invalid index')
            if type(key[1]) is str and type(key[0]) is int:
                return self.index_regime_purpose[key[0]][key[1]]
            elif type(key[0]) is str and type(key[1]) is int:
                return self.index_regime_purpose[key[1]][key[0]]
            else:
                raise IndexError('Invalid index')
        else:
            raise IndexError('Invalid index')

    def __setitem__(self, key, value):
        _type = type(key)
        if _type is str:
            value = np.array(value, dtype=bool, ndmin=1)
            k_params = self.k_params
            self.k_parameters[key] = value.size + np.sum(value) * (self.k_regimes - 1)
            self.k_params += self.k_parameters[key]
            self.switching[key] = value
            self.slices_purpose[key] = np.s_[k_params:self.k_params]
            for j in range(self.k_regimes):
                self.relative_index_regime_purpose[j][key] = []
                self.index_regime_purpose[j][key] = []
            offset = 0
            for i in range(value.size):
                switching = value[i]
                for j in range(self.k_regimes):
                    if not switching:
                        self.relative_index_regime_purpose[j][key].append(offset)
                    else:
                        self.relative_index_regime_purpose[j][key].append(offset + j)
                offset += 1 if not switching else self.k_regimes
            for j in range(self.k_regimes):
                offset = 0
                indices = []
                for k, v in self.relative_index_regime_purpose[j].items():
                    v = (np.r_[v] + offset).tolist()
                    self.index_regime_purpose[j][k] = v
                    indices.append(v)
                    offset += self.k_parameters[k]
                self.index_regime[j] = np.concatenate(indices).astype(int)
        else:
            raise IndexError('Invalid index')