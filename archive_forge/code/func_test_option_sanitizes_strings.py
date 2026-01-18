from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_option_sanitizes_strings():
    assert Option('float32').ty == dshape('float32').measure