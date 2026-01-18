from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm

    Test Series.map with a dictionary subclass that defines __missing__,
    i.e. sets a default value (GH #15999).
    