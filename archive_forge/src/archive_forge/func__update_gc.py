from matplotlib.backend_bases import RendererBase
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import transforms as mtransforms
from matplotlib.path import Path
import numpy as np
def _update_gc(self, gc, new_gc_dict):
    """
        Update the given GraphicsContext with the given dict of properties.

        The keys in the dictionary are used to identify the appropriate
        ``set_`` method on the *gc*.
        """
    new_gc_dict = new_gc_dict.copy()
    dashes = new_gc_dict.pop('dashes', None)
    if dashes:
        gc.set_dashes(**dashes)
    for k, v in new_gc_dict.items():
        set_method = getattr(gc, 'set_' + k, None)
        if not callable(set_method):
            raise AttributeError(f'Unknown property {k}')
        set_method(v)
    return gc