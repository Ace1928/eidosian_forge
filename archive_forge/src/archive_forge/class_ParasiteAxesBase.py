from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Bbox
from .mpl_axes import Axes
class ParasiteAxesBase:

    def __init__(self, parent_axes, aux_transform=None, *, viewlim_mode=None, **kwargs):
        self._parent_axes = parent_axes
        self.transAux = aux_transform
        self.set_viewlim_mode(viewlim_mode)
        kwargs['frameon'] = False
        super().__init__(parent_axes.figure, parent_axes._position, **kwargs)

    def clear(self):
        super().clear()
        martist.setp(self.get_children(), visible=False)
        self._get_lines = self._parent_axes._get_lines
        self._parent_axes.callbacks._connect_picklable('xlim_changed', self._sync_lims)
        self._parent_axes.callbacks._connect_picklable('ylim_changed', self._sync_lims)

    def pick(self, mouseevent):
        super().pick(mouseevent)
        for a in self.get_children():
            if hasattr(mouseevent.inaxes, 'parasites') and self in mouseevent.inaxes.parasites:
                a.pick(mouseevent)

    def _set_lim_and_transforms(self):
        if self.transAux is not None:
            self.transAxes = self._parent_axes.transAxes
            self.transData = self.transAux + self._parent_axes.transData
            self._xaxis_transform = mtransforms.blended_transform_factory(self.transData, self.transAxes)
            self._yaxis_transform = mtransforms.blended_transform_factory(self.transAxes, self.transData)
        else:
            super()._set_lim_and_transforms()

    def set_viewlim_mode(self, mode):
        _api.check_in_list([None, 'equal', 'transform'], mode=mode)
        self._viewlim_mode = mode

    def get_viewlim_mode(self):
        return self._viewlim_mode

    def _sync_lims(self, parent):
        viewlim = parent.viewLim.frozen()
        mode = self.get_viewlim_mode()
        if mode is None:
            pass
        elif mode == 'equal':
            self.viewLim.set(viewlim)
        elif mode == 'transform':
            self.viewLim.set(viewlim.transformed(self.transAux.inverted()))
        else:
            _api.check_in_list([None, 'equal', 'transform'], mode=mode)