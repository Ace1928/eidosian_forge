from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_timedelta_nano():
    dshape('timedelta[unit="ns"]').measure.unit == 'ns'