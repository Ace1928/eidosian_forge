from __future__ import annotations
import collections.abc
import copy
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
import numpy as np
import pandas as pd
from xarray.core import formatting, nputils, utils
from xarray.core.indexing import (
from xarray.core.utils import (
def _is_nested_tuple(possible_tuple):
    return isinstance(possible_tuple, tuple) and any((isinstance(value, (tuple, list, slice)) for value in possible_tuple))