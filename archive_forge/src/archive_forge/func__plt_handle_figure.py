from __future__ import division
import numpy as np
from pygsp import utils
def _plt_handle_figure(plot):

    def inner(obj, *args, **kwargs):
        plt = _import_plt()
        if 'ax' not in kwargs.keys():
            fig = plt.figure()
            global _plt_figures
            _plt_figures.append(fig)
            if hasattr(obj, 'coords') and obj.coords.ndim == 2 and (obj.coords.shape[1] == 3):
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
            kwargs.update(ax=ax)
        save_as = kwargs.pop('save_as', None)
        plot_name = kwargs.pop('plot_name', '')
        plot(obj, *args, **kwargs)
        kwargs['ax'].set_title(plot_name)
        try:
            if save_as is not None:
                fig.savefig(save_as + '.png')
                fig.savefig(save_as + '.pdf')
            else:
                fig.show(warn=False)
        except NameError:
            pass
    return inner