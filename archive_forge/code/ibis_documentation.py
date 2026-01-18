import sys
from collections.abc import Iterable
from functools import lru_cache
import numpy as np
from packaging.version import Version
from .. import util
from ..element import Element
from ..ndmapping import NdMapping, item_check, sorted_context
from . import pandas
from .interface import Interface
from .util import cached

        Given a dataset object and data in the appropriate format for
        the interface, return a simple scalar.
        