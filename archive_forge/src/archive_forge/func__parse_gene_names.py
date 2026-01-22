from .. import sanitize
from .. import utils
import numpy as np
import pandas as pd
import scipy.sparse as sp
import warnings
def _parse_gene_names(header, data):
    header = _parse_header(header, data.shape[1], header_type='gene_names')
    if header is None:
        try:
            return data.columns
        except AttributeError:
            pass
    return header