from collections import OrderedDict
import dateutil.parser
import itertools
import logging
import numpy as np
from matplotlib import _api, ticker, units
@staticmethod
def _validate_unit(unit):
    if not hasattr(unit, '_mapping'):
        raise ValueError(f'Provided unit "{unit}" is not valid for a categorical converter, as it does not have a _mapping attribute.')