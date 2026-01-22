import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class CustomDataFrame(DataFrame):
    """
            Subclasses pandas DF, fills DF with simulation results, adds some
            custom plotting functions.
            """

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)

    @property
    def _constructor(self):
        return CustomDataFrame
    _constructor_sliced = CustomSeries

    def custom_frame_function(self):
        return 'OK'