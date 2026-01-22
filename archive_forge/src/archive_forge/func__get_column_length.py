from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def _get_column_length(data):
    try:
        return data.shape[1]
    except (IndexError, AttributeError):
        return len(data)