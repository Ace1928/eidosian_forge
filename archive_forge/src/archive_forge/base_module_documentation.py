import time
import logging
import warnings
import numpy as np
from .. import metric
from .. import ndarray
from ..context import cpu
from ..model import BatchEndParam
from ..initializer import Uniform
from ..io import DataDesc, DataIter, DataBatch
from ..base import _as_list
Gets the symbol associated with this module.

        Except for `Module`, for other types of modules (e.g. `BucketingModule`), this
        property might not be a constant throughout its life time. Some modules might
        not even be associated with any symbols.
        