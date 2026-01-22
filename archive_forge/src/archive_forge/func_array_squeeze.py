from traitlets import TraitError, TraitType
import numpy as np
import pandas as pd
import warnings
import datetime as dt
import six
def array_squeeze(trait, value):
    if len(value.shape) > 1:
        return np.squeeze(value)
    else:
        return value