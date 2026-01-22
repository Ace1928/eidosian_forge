from numbers import Real
from matplotlib import _api
from matplotlib.axes import Axes
class AxesX(_Base):
    """
    Scaled size whose relative part corresponds to the data width
    of the *axes* multiplied by the *aspect*.
    """

    def __init__(self, axes, aspect=1.0, ref_ax=None):
        self._axes = axes
        self._aspect = aspect
        if aspect == 'axes' and ref_ax is None:
            raise ValueError("ref_ax must be set when aspect='axes'")
        self._ref_ax = ref_ax

    def get_size(self, renderer):
        l1, l2 = self._axes.get_xlim()
        if self._aspect == 'axes':
            ref_aspect = _get_axes_aspect(self._ref_ax)
            aspect = ref_aspect / _get_axes_aspect(self._axes)
        else:
            aspect = self._aspect
        rel_size = abs(l2 - l1) * aspect
        abs_size = 0.0
        return (rel_size, abs_size)