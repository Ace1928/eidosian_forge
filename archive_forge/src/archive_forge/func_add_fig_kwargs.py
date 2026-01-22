from __future__ import annotations
import importlib
import math
from functools import wraps
from string import ascii_letters
from typing import TYPE_CHECKING, Literal
import matplotlib.pyplot as plt
import numpy as np
import palettable.colorbrewer.diverging
from matplotlib import cm, colors
from pymatgen.core import Element
def add_fig_kwargs(func):
    """Decorator that adds keyword arguments for functions returning matplotlib
    figures.

    The function should return either a matplotlib figure or None to signal
    some sort of error/unexpected event.
    See doc string below for the list of supported options.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        title = kwargs.pop('title', None)
        size_kwargs = kwargs.pop('size_kwargs', None)
        show = kwargs.pop('show', True)
        savefig = kwargs.pop('savefig', None)
        tight_layout = kwargs.pop('tight_layout', False)
        ax_grid = kwargs.pop('ax_grid', None)
        ax_annotate = kwargs.pop('ax_annotate', None)
        fig_close = kwargs.pop('fig_close', False)
        fig = func(*args, **kwargs)
        if fig is None:
            return fig
        if title is not None:
            fig.suptitle(title)
        if size_kwargs is not None:
            fig.set_size_inches(size_kwargs.pop('w'), size_kwargs.pop('h'), **size_kwargs)
        if ax_grid is not None:
            for ax in fig.axes:
                ax.grid(bool(ax_grid))
        if ax_annotate:
            tags = ascii_letters
            if len(fig.axes) > len(tags):
                tags = (1 + len(ascii_letters) // len(fig.axes)) * ascii_letters
            for ax, tag in zip(fig.axes, tags):
                ax.annotate(f'({tag})', xy=(0.05, 0.95), xycoords='axes fraction')
        if tight_layout:
            try:
                fig.tight_layout()
            except Exception as exc:
                print('Ignoring Exception raised by fig.tight_layout\n', str(exc))
        if savefig:
            fig.savefig(savefig)
        if show:
            plt.show()
        if fig_close:
            plt.close(fig=fig)
        return fig
    doc_str = '\n\n\n        Keyword arguments controlling the display of the figure:\n\n        ================  ====================================================\n        kwargs            Meaning\n        ================  ====================================================\n        title             Title of the plot (Default: None).\n        show              True to show the figure (default: True).\n        savefig           "abc.png" or "abc.eps" to save the figure to a file.\n        size_kwargs       Dictionary with options passed to fig.set_size_inches\n                          e.g. size_kwargs=dict(w=3, h=4)\n        tight_layout      True to call fig.tight_layout (default: False)\n        ax_grid           True (False) to add (remove) grid from all axes in fig.\n                          Default: None i.e. fig is left unchanged.\n        ax_annotate       Add labels to  subplots e.g. (a), (b).\n                          Default: False\n        fig_close         Close figure. Default: False.\n        ================  ====================================================\n\n'
    if wrapper.__doc__ is not None:
        wrapper.__doc__ += f'\n{doc_str}'
    else:
        wrapper.__doc__ = doc_str
    return wrapper