from . import _catboost
from .core import Pool, CatBoostError, ARRAY_TYPES, PATH_TYPES, fspath, _update_params_quantize_part, _process_synonyms
from collections import defaultdict
from contextlib import contextmanager
import sys
import numpy as np
import warnings
@contextmanager
def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        warnings.warn('To draw plots you should install matplotlib.')
        raise ImportError(str(e))
    yield plt