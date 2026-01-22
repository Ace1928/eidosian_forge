from traitlets import TraitError, TraitType
import numpy as np
import pandas as pd
import warnings
import datetime as dt
import six
def _array_equal(a, b):
    """Really tests if arrays are equal, where nan == nan == True"""
    try:
        return np.allclose(a, b, 0, 0, equal_nan=True)
    except (TypeError, ValueError):
        return False