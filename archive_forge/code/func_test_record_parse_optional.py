from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_record_parse_optional(b, c):
    assert all((isinstance(ty, Option) for ty in c.types))
    assert [ty for ty in b.types] == [oty.ty for oty in c.types]