from __future__ import annotations
import contextlib
import copy
import io
import pickle as pkl
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import BaseOffset
from pandas import Index
from pandas.core.arrays import (
from pandas.core.internals import BlockManager

    Temporarily patch pickle to use our unpickler.
    