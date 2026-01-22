from __future__ import annotations
from typing import (
import numpy as np
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
import pandas as pd
from pandas.core.shared_docs import _shared_docs
from pandas.io.excel._base import BaseExcelReader

        Find and decode OpenDocument text:s tags that represent
        a run length encoded sequence of space characters.
        