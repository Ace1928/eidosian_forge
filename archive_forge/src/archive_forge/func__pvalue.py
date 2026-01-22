import numpy as np
import pandas as pd
from statsmodels.graphics.utils import maybe_name_or_idx
def _pvalue(vec):
    return 2 * min(sum(vec > 0), sum(vec < 0)) / float(len(vec))