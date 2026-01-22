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
def _in_unstable_openblas_configuration():
    """Return True if in an unstable configuration for OpenBLAS"""
    import numpy
    import scipy
    modules_info = threadpool_info()
    open_blas_used = any((info['internal_api'] == 'openblas' for info in modules_info))
    if not open_blas_used:
        return False
    openblas_arm64_stable_version = parse_version('0.3.16')
    for info in modules_info:
        if info['internal_api'] != 'openblas':
            continue
        openblas_version = info.get('version')
        openblas_architecture = info.get('architecture')
        if openblas_version is None or openblas_architecture is None:
            return True
        if openblas_architecture == 'neoversen1' and parse_version(openblas_version) < openblas_arm64_stable_version:
            return True
    return False