from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def _check_NaCl(cneqsys, guesses, cases=-1, **kwargs):
    _init_final = [([1, 1, 1], [2, 2, 0]), ([1, 1, 0], [1, 1, 0]), ([3, 3, 3], [2, 2, 4]), ([2, 2, 0], [2, 2, 0]), ([2 + 1e-08, 2 + 1e-08, 0], [2, 2, 1e-08]), ([3, 3, 0], [2, 2, 1]), ([0, 0, 3], [2, 2, 1]), ([0, 0, 2], [2, 2, 0]), ([2, 2, 2], [2, 2, 2])]
    info_dicts = []
    for init, final in _init_final[:cases]:
        print(init)
        for guess in guesses:
            print(guess)
            if guess is None:
                guess = init
            x, info_dict = cneqsys.solve(guess, init + [4], solver='scipy', **kwargs)
            assert info_dict['success'] and np.allclose(x, final)
            info_dicts.append(info_dict)
    return info_dicts