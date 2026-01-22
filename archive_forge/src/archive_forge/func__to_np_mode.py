import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
def _to_np_mode(mode):
    """Convert padding modes from `ndi.correlate` to `np.pad`."""
    mode_translation_dict = dict(nearest='edge', reflect='symmetric', mirror='reflect')
    if mode in mode_translation_dict:
        mode = mode_translation_dict[mode]
    return mode