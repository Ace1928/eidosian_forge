import copy
import gc
import inspect
import runpy
import sys
import threading
from collections import namedtuple
from enum import Enum
from functools import wraps, partial
from typing import List, Any, ClassVar, Optional, Sequence, Tuple, Union, Dict, Set
import unittest
import os
import torch
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM, TEST_MKL, \
from torch.testing._internal.common_cuda import _get_torch_cuda_version, \
from torch.testing._internal.common_dtype import get_all_dtypes
def _dtype_test_suffix(dtypes):
    """ Returns the test suffix for a dtype, sequence of dtypes, or None. """
    if isinstance(dtypes, (list, tuple)):
        if len(dtypes) == 0:
            return ''
        return '_' + '_'.join((dtype_name(d) for d in dtypes))
    elif dtypes:
        return f'_{dtype_name(dtypes)}'
    else:
        return ''