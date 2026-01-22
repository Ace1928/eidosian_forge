from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class NonDictMappingWithMissing(non_dict_mapping_subclass):

    def __missing__(self, key):
        return 'missing'