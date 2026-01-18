import numpy as np
import pytest
from matplotlib.transforms import Bbox
from mpl_toolkits.axisartist.grid_finder import (
def test_pretty_print_format():
    locator = MaxNLocator()
    locs, nloc, factor = locator(0, 100)
    fmt = FormatterPrettyPrint()
    assert fmt('left', None, locs) == ['$\\mathdefault{%d}$' % (l,) for l in locs]