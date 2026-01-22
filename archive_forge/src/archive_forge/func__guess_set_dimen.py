import sys
import re
import copy
import logging
from pyomo.common.log import is_debug_set
from pyomo.common.collections import Bunch, OrderedDict
from pyomo.common.errors import ApplicationError
from pyomo.dataportal.parse_datacmds import parse_data_commands, _re_number
from pyomo.dataportal.factory import DataManagerFactory, UnknownDataManager
from pyomo.core.base.set import UnknownSetDimen
from pyomo.core.base.util import flatten_tuple
def _guess_set_dimen(index):
    d = 0
    for subset in index.subsets():
        sub_d = subset.dimen
        if sub_d is UnknownSetDimen:
            for domain_subset in subset.domain.subsets():
                sub_d = domain_subset.domain.dimen
                if sub_d in (UnknownSetDimen, None):
                    d += 1
                else:
                    d += sub_d
        elif sub_d is None:
            return None
        else:
            d += sub_d
    return d