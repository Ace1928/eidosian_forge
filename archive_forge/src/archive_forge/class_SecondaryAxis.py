import numbers
import numpy as np
from matplotlib import _api, _docstring
import matplotlib.ticker as mticker
from matplotlib.axes._base import _AxesBase, _TransformedBoundsLocator
from matplotlib.axis import Axis
class SecondaryAxis(_AxesBase):
    """
    General class to hold a Secondary_X/Yaxis.
    """

    def __init__(self, parent, orientation, location, functions, **kwargs):
        """
        See `.secondary_xaxis` and `.secondary_yaxis` for the doc string.
        While there is no need for this to be private, it should really be
        called by those higher level functions.
        """
        _api.check_in_list(['x', 'y'], orientation=orientation)
        self._functions = functions
        self._parent = parent
        self._orientation = orientation
        self._ticks_set = False
        if self._orientation == 'x':
            super().__init__(self._parent.figure, [0, 1.0, 1, 0.0001], **kwargs)
            self._axis = self.xaxis
            self._locstrings = ['top', 'bottom']
            self._otherstrings = ['left', 'right']
        else:
            super().__init__(self._parent.figure, [0, 1.0, 0.0001, 1], **kwargs)
            self._axis = self.yaxis
            self._locstrings = ['right', 'left']
            self._otherstrings = ['top', 'bottom']
        self._parentscale = None
        self.set_location(location)
        self.set_functions(functions)
        otheraxis = self.yaxis if self._orientation == 'x' else self.xaxis
        otheraxis.set_major_locator(mticker.NullLocator())
        otheraxis.set_ticks_position('none')
        self.spines[self._otherstrings].set_visible(False)
        self.spines[self._locstrings].set_visible(True)
        if self._pos < 0.5:
            self._locstrings = self._locstrings[::-1]
        self.set_alignment(self._locstrings[0])

    def set_alignment(self, align):
        """
        Set if axes spine and labels are drawn at top or bottom (or left/right)
        of the axes.

        Parameters
        ----------
        align : {'top', 'bottom', 'left', 'right'}
            Either 'top' or 'bottom' for orientation='x' or
            'left' or 'right' for orientation='y' axis.
        """
        _api.check_in_list(self._locstrings, align=align)
        if align == self._locstrings[1]:
            self._locstrings = self._locstrings[::-1]
        self.spines[self._locstrings[0]].set_visible(True)
        self.spines[self._locstrings[1]].set_visible(False)
        self._axis.set_ticks_position(align)
        self._axis.set_label_position(align)

    def set_location(self, location):
        """
        Set the vertical or horizontal location of the axes in
        parent-normalized coordinates.

        Parameters
        ----------
        location : {'top', 'bottom', 'left', 'right'} or float
            The position to put the secondary axis.  Strings can be 'top' or
            'bottom' for orientation='x' and 'right' or 'left' for
            orientation='y'. A float indicates the relative position on the
            parent axes to put the new axes, 0.0 being the bottom (or left)
            and 1.0 being the top (or right).
        """
        if isinstance(location, str):
            _api.check_in_list(self._locstrings, location=location)
            self._pos = 1.0 if location in ('top', 'right') else 0.0
        elif isinstance(location, numbers.Real):
            self._pos = location
        else:
            raise ValueError(f'location must be {self._locstrings[0]!r}, {self._locstrings[1]!r}, or a float, not {location!r}')
        self._loc = location
        if self._orientation == 'x':
            bounds = [0, self._pos, 1.0, 1e-10]
        else:
            bounds = [self._pos, 0, 1e-10, 1]
        self.set_axes_locator(_TransformedBoundsLocator(bounds, self._parent.transAxes))

    def apply_aspect(self, position=None):
        self._set_lims()
        super().apply_aspect(position)

    @_docstring.copy(Axis.set_ticks)
    def set_ticks(self, ticks, labels=None, *, minor=False, **kwargs):
        ret = self._axis.set_ticks(ticks, labels, minor=minor, **kwargs)
        self.stale = True
        self._ticks_set = True
        return ret

    def set_functions(self, functions):
        """
        Set how the secondary axis converts limits from the parent axes.

        Parameters
        ----------
        functions : 2-tuple of func, or `Transform` with an inverse.
            Transform between the parent axis values and the secondary axis
            values.

            If supplied as a 2-tuple of functions, the first function is
            the forward transform function and the second is the inverse
            transform.

            If a transform is supplied, then the transform must have an
            inverse.
        """
        if isinstance(functions, tuple) and len(functions) == 2 and callable(functions[0]) and callable(functions[1]):
            self._functions = functions
        elif functions is None:
            self._functions = (lambda x: x, lambda x: x)
        else:
            raise ValueError('functions argument of secondary axes must be a two-tuple of callable functions with the first function being the transform and the second being the inverse')
        self._set_scale()

    def draw(self, renderer):
        """
        Draw the secondary axes.

        Consults the parent axes for its limits and converts them
        using the converter specified by
        `~.axes._secondary_axes.set_functions` (or *functions*
        parameter when axes initialized.)
        """
        self._set_lims()
        self._set_scale()
        super().draw(renderer)

    def _set_scale(self):
        """
        Check if parent has set its scale
        """
        if self._orientation == 'x':
            pscale = self._parent.xaxis.get_scale()
            set_scale = self.set_xscale
        else:
            pscale = self._parent.yaxis.get_scale()
            set_scale = self.set_yscale
        if pscale == self._parentscale:
            return
        if self._ticks_set:
            ticks = self._axis.get_ticklocs()
        set_scale('functionlog' if pscale == 'log' else 'function', functions=self._functions[::-1])
        if self._ticks_set:
            self._axis.set_major_locator(mticker.FixedLocator(ticks))
        self._parentscale = pscale

    def _set_lims(self):
        """
        Set the limits based on parent limits and the convert method
        between the parent and this secondary axes.
        """
        if self._orientation == 'x':
            lims = self._parent.get_xlim()
            set_lim = self.set_xlim
        else:
            lims = self._parent.get_ylim()
            set_lim = self.set_ylim
        order = lims[0] < lims[1]
        lims = self._functions[0](np.array(lims))
        neworder = lims[0] < lims[1]
        if neworder != order:
            lims = lims[::-1]
        set_lim(lims)

    def set_aspect(self, *args, **kwargs):
        """
        Secondary axes cannot set the aspect ratio, so calling this just
        sets a warning.
        """
        _api.warn_external("Secondary axes can't set the aspect ratio")

    def set_color(self, color):
        """
        Change the color of the secondary axes and all decorators.

        Parameters
        ----------
        color : color
        """
        axis = self._axis_map[self._orientation]
        axis.set_tick_params(colors=color)
        for spine in self.spines.values():
            if spine.axis is axis:
                spine.set_color(color)
        axis.label.set_color(color)