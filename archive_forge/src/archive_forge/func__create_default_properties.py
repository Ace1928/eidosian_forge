from statsmodels.compat.python import lrange, lzip
from itertools import product
import numpy as np
from numpy import array, cumsum, iterable, r_
from pandas import DataFrame
from statsmodels.graphics import utils
def _create_default_properties(data):
    """"Create the default properties of the mosaic given the data
    first it will varies the color hue (first category) then the color
    saturation (second category) and then the color value
    (third category).  If a fourth category is found, it will put
    decoration on the rectangle.  Does not manage more than four
    level of categories
    """
    categories_levels = _categories_level(list(data.keys()))
    Nlevels = len(categories_levels)
    L = len(categories_levels[0])
    hue = np.linspace(0.0, 1.0, L + 2)[:-2]
    L = len(categories_levels[1]) if Nlevels > 1 else 1
    saturation = np.linspace(0.5, 1.0, L + 1)[:-1]
    L = len(categories_levels[2]) if Nlevels > 2 else 1
    value = np.linspace(0.5, 1.0, L + 1)[:-1]
    L = len(categories_levels[3]) if Nlevels > 3 else 1
    hatch = ['', '/', '-', '|', '+'][:L + 1]
    hue = lzip(list(hue), categories_levels[0])
    saturation = lzip(list(saturation), categories_levels[1] if Nlevels > 1 else [''])
    value = lzip(list(value), categories_levels[2] if Nlevels > 2 else [''])
    hatch = lzip(list(hatch), categories_levels[3] if Nlevels > 3 else [''])
    properties = {}
    for h, s, v, t in product(hue, saturation, value, hatch):
        hv, hn = h
        sv, sn = s
        vv, vn = v
        tv, tn = t
        level = (hn,) + ((sn,) if sn else tuple())
        level = level + ((vn,) if vn else tuple())
        level = level + ((tn,) if tn else tuple())
        hsv = array([hv, sv, vv])
        prop = {'color': _single_hsv_to_rgb(hsv), 'hatch': tv, 'lw': 0}
        properties[level] = prop
    return properties