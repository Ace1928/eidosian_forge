import math
from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
def value_at(self, a, xi, yi):
    """
        Set up for RK4 function, based on Bokeh's streamline code
        """
    if isinstance(xi, np.ndarray):
        self.x = xi.astype(int)
        self.y = yi.astype(int)
    else:
        self.val_x = int(xi)
        self.val_y = int(yi)
    a00 = a[self.val_y, self.val_x]
    a01 = a[self.val_y, self.val_x + 1]
    a10 = a[self.val_y + 1, self.val_x]
    a11 = a[self.val_y + 1, self.val_x + 1]
    xt = xi - self.val_x
    yt = yi - self.val_y
    a0 = a00 * (1 - xt) + a01 * xt
    a1 = a10 * (1 - xt) + a11 * xt
    return a0 * (1 - yt) + a1 * yt