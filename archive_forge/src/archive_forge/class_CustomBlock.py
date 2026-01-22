import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
import pandas.util._test_decorators as td
import pandas as pd
from pandas.core.internals import BlockManager
from pandas.core.internals.blocks import ExtensionBlock
class CustomBlock(ExtensionBlock):
    _holder = np.ndarray

    @property
    def _can_hold_na(self) -> bool:
        return False