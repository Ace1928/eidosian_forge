from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
@pytest.mark.skip
class TestJoinSindex:

    def setup_method(self):
        nybb_filename = geopandas.datasets.get_path('nybb')
        self.boros = read_file(nybb_filename)

    def test_merge_geo(self):
        tree = self.boros.sindex
        hits = tree.intersection((1012821.8, 229228.26))
        res = [self.boros.iloc[hit]['BoroName'] for hit in hits]
        assert res == ['Bronx', 'Queens']
        first = self.boros[self.boros['BoroCode'] < 3]
        tree = first.sindex
        hits = tree.intersection((1012821.8, 229228.26))
        res = [first.iloc[hit]['BoroName'] for hit in hits]
        assert res == ['Bronx']
        second = self.boros[self.boros['BoroCode'] >= 3]
        tree = second.sindex
        hits = tree.intersection((1012821.8, 229228.26))
        res = ([second.iloc[hit]['BoroName'] for hit in hits],)
        assert res == ['Queens']
        merged = first.merge(second, how='outer')
        assert len(merged) == 5
        assert merged.sindex.size == 5
        tree = merged.sindex
        hits = tree.intersection((1012821.8, 229228.26))
        res = [merged.iloc[hit]['BoroName'] for hit in hits]
        assert res == ['Bronx', 'Queens']