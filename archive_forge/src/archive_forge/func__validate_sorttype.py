import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def _validate_sorttype(sort_type, list):
    sort_type = sort_type.lower()
    if sort_type not in list:
        raise YappiError(f"Invalid SortType parameter: '{sort_type}'")
    return sort_type