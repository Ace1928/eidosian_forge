import pytest
from pandas import (
def test_overlaps_disjoint(self, start_shift, closed, other_closed):
    start, shift = start_shift
    interval1 = Interval(start, start + shift, other_closed)
    interval2 = Interval(start + 2 * shift, start + 3 * shift, closed)
    assert not interval1.overlaps(interval2)