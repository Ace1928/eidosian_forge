from datetime import datetime
from typing import (
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.frame import DataFrame
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
Concatenate dataframes

        :param dfs: the dataframes to concatenate
        :return: the concatenated dataframe
        