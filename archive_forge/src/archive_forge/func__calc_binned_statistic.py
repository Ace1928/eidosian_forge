import builtins
from warnings import catch_warnings, simplefilter
import numpy as np
from operator import index
from collections import namedtuple
def _calc_binned_statistic(Vdim, bin_numbers, result, values, stat_func):
    unique_bin_numbers = np.unique(bin_numbers)
    for vv in builtins.range(Vdim):
        bin_map = _create_binned_data(bin_numbers, unique_bin_numbers, values, vv)
        for i in unique_bin_numbers:
            stat = stat_func(np.array(bin_map[i]))
            if np.iscomplexobj(stat) and (not np.iscomplexobj(result)):
                raise ValueError('The statistic function returns complex ')
            result[vv, i] = stat