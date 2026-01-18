import numpy as np
from matplotlib import ticker as mticker
from matplotlib.transforms import Bbox, Transform
def transform_xy(self, x, y):
    return self._aux_transform.transform(np.column_stack([x, y])).T