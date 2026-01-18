from __future__ import annotations
from collections.abc import (
from typing import (
import numpy as np
from pandas._libs import algos as libalgos
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import BaseMaskedDtype

            Helper function to concat `current_indexer` and `other_indexer`
            depending on `method`
            