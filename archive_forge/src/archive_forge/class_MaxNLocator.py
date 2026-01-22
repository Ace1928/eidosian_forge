import numpy as np
from matplotlib import ticker as mticker
from matplotlib.transforms import Bbox, Transform
class MaxNLocator(mticker.MaxNLocator):

    def __init__(self, nbins=10, steps=None, trim=True, integer=False, symmetric=False, prune=None):
        super().__init__(nbins, steps=steps, integer=integer, symmetric=symmetric, prune=prune)
        self.create_dummy_axis()

    def __call__(self, v1, v2):
        locs = super().tick_values(v1, v2)
        return (np.array(locs), len(locs), 1)