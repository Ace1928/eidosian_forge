import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
def to_pydict(self, *args, **kwargs):
    """
        Convert the Table to a `dict` or `OrderedDict`.

        Returns:
            `dict`
        """
    return self.table.to_pydict(*args, **kwargs)