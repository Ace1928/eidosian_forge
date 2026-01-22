from .. import utils
from .._lazyload import matplotlib as mpl
from .._lazyload import mpl_toolkits
import numpy as np
import platform
def _get_figure(ax=None, figsize=None, subplot_kw=None):
    if subplot_kw is None:
        subplot_kw = {}
    if ax is None:
        if 'projection' in subplot_kw and subplot_kw['projection'] == '3d':
            mpl_toolkits.mplot3d.Axes3D
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=subplot_kw)
        show_fig = True
    else:
        try:
            fig = ax.get_figure()
        except AttributeError as e:
            if not isinstance(ax, mpl.axes.Axes):
                raise TypeError('Expected ax as a matplotlib.axes.Axes. Got {}'.format(type(ax)))
            else:
                raise e
        if 'projection' in subplot_kw:
            if subplot_kw['projection'] == '3d' and (not isinstance(ax, mpl_toolkits.mplot3d.Axes3D)):
                raise TypeError("Expected ax with projection='3d'. Got 2D axis instead.")
        show_fig = False
    return (fig, ax, show_fig)