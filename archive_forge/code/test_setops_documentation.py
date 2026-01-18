from datetime import (
from hypothesis import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm

    Check that we either have a RangeIndex or that this index *cannot*
    be represented as a RangeIndex.
    