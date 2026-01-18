from collections.abc import Mapping
import functools
import numpy as np
from numpy import ma
import matplotlib as mpl
from matplotlib import _api, colors, cbook, scale
from matplotlib._cm import datad
from matplotlib._cm_listed import cmaps as cmaps_listed
@_api.deprecated('3.7', alternative='``matplotlib.colormaps.unregister(name)``')
def unregister_cmap(name):
    """
    Remove a colormap recognized by :func:`get_cmap`.

    You may not remove built-in colormaps.

    If the named colormap is not registered, returns with no error, raises
    if you try to de-register a default colormap.

    .. warning::

      Colormap names are currently a shared namespace that may be used
      by multiple packages. Use `unregister_cmap` only if you know you
      have registered that name before. In particular, do not
      unregister just in case to clean the name before registering a
      new colormap.

    Parameters
    ----------
    name : str
        The name of the colormap to be un-registered

    Returns
    -------
    ColorMap or None
        If the colormap was registered, return it if not return `None`

    Raises
    ------
    ValueError
       If you try to de-register a default built-in colormap.
    """
    cmap = _colormaps.get(name, None)
    _colormaps.unregister(name)
    return cmap