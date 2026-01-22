from .. import utils
from .._lazyload import matplotlib as mpl
from .._lazyload import mpl_toolkits
import numpy as np
import platform
def _is_color_array(c):
    if c is None:
        return False
    else:
        try:
            c_rgb = mpl.colors.to_rgba_array(c)
            if np.any(c_rgb > 1):
                for i in np.argwhere(c_rgb > 1).flatten():
                    if isinstance(c[i], str):
                        return False
            return True
        except ValueError:
            return False