from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def _get_row_length(data):
    try:
        return data.shape[0]
    except (IndexError, AttributeError):
        return len(data)