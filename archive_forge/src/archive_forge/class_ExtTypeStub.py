from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
class ExtTypeStub(pd.api.extensions.ExtensionArray):

    def __len__(self) -> int:
        return 2

    def __getitem__(self, ix):
        return [ix == 1, ix == 0]

    @property
    def dtype(self):
        return DtypeStub()