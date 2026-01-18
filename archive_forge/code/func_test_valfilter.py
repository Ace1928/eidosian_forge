from collections import defaultdict as _defaultdict
from collections.abc import Mapping
import os
from toolz.dicttoolz import (merge, merge_with, valmap, keymap, update_in,
from toolz.functoolz import identity
from toolz.utils import raises
def test_valfilter(self):
    D, kw = (self.D, self.kw)
    assert valfilter(iseven, D({1: 2, 2: 3}), **kw) == D({1: 2})