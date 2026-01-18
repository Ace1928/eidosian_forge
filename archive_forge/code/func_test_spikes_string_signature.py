import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_spikes_string_signature(self):
    spikes = Spikes([], 'a')
    self.assertEqual(spikes.kdims, [Dimension('a')])