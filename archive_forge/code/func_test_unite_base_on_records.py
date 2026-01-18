from collections import OrderedDict
from itertools import starmap
from types import MappingProxyType
from warnings import catch_warnings, simplefilter
import numpy as np
import pytest
from datashader.datashape.discovery import (
from datashader.datashape.coretypes import (
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape import dshape
from datetime import date, time, datetime, timedelta
def test_unite_base_on_records():
    dshapes = [dshape('{name: string, amount: int32}'), dshape('{name: string, amount: int32}')]
    assert unite_base(dshapes) == dshape('2 * {name: string, amount: int32}')
    dshapes = [Null(), dshape('{name: string, amount: int32}')]
    assert unite_base(dshapes) == dshape('2 * ?{name: string, amount: int32}')
    dshapes = [dshape('{name: string, amount: int32}'), dshape('{name: string, amount: int64}')]
    assert unite_base(dshapes) == dshape('2 * {name: string, amount: int64}')