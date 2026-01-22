from .. import sanitize
from .. import utils
import numpy as np
import pandas as pd
import scipy.sparse as sp
import warnings
def _parse_cell_names(header, data):
    header = _parse_header(header, data.shape[0], header_type='cell_names')
    if header is None:
        try:
            return data.index
        except AttributeError:
            pass
    return header