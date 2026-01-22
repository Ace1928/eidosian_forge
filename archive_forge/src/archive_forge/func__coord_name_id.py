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
@property
def _coord_name_id(self) -> dict[Any, int]:
    if self.__coord_name_id is None:
        self.__coord_name_id = {k: id(idx) for k, idx in self._indexes.items()}
    return self.__coord_name_id