import math
import numbers
import platform
import struct
import timeit
import warnings
from collections.abc import Sequence
from contextlib import contextmanager, suppress
from itertools import compress, islice
import numpy as np
from scipy.sparse import issparse
from .. import get_config
from ..exceptions import DataConversionWarning
from . import _joblib, metadata_routing
from ._bunch import Bunch
from ._estimator_html_repr import estimator_html_repr
from ._param_validation import Integral, Interval, validate_params
from .class_weight import compute_class_weight, compute_sample_weight
from .deprecation import deprecated
from .discovery import all_estimators
from .fixes import parse_version, threadpool_info
from .murmurhash import murmurhash3_32
from .validation import (
@contextmanager
def _print_elapsed_time(source, message=None):
    """Log elapsed time to stdout when the context is exited.

    Parameters
    ----------
    source : str
        String indicating the source or the reference of the message.

    message : str, default=None
        Short message. If None, nothing will be printed.

    Returns
    -------
    context_manager
        Prints elapsed time upon exit if verbose.
    """
    if message is None:
        yield
    else:
        start = timeit.default_timer()
        yield
        print(_message_with_time(source, message, timeit.default_timer() - start))