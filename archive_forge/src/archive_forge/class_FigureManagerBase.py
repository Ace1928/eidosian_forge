from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum, IntEnum
import functools
import importlib
import inspect
import io
import itertools
import logging
import os
import sys
import time
import weakref
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib import (
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_managers import ToolManager
from matplotlib.cbook import _setattr_cm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib._enums import JoinStyle, CapStyle
class FigureManagerBase:
    """
    A backend-independent abstraction of a figure container and controller.

    The figure manager is used by pyplot to interact with the window in a
    backend-independent way. It's an adapter for the real (GUI) framework that
    represents the visual figure on screen.

    The figure manager is connected to a specific canvas instance, which in turn
    is connected to a specific figure instance. To access a figure manager for
    a given figure in user code, you typically use ``fig.canvas.manager``.

    GUI backends derive from this class to translate common operations such
    as *show* or *resize* to the GUI-specific code. Non-GUI backends do not
    support these operations and can just use the base class.

    This following basic operations are accessible:

    **Window operations**

    - `~.FigureManagerBase.show`
    - `~.FigureManagerBase.destroy`
    - `~.FigureManagerBase.full_screen_toggle`
    - `~.FigureManagerBase.resize`
    - `~.FigureManagerBase.get_window_title`
    - `~.FigureManagerBase.set_window_title`

    **Key and mouse button press handling**

    The figure manager sets up default key and mouse button press handling by
    hooking up the `.key_press_handler` to the matplotlib event system. This
    ensures the same shortcuts and mouse actions across backends.

    **Other operations**

    Subclasses will have additional attributes and functions to access
    additional functionality. This is of course backend-specific. For example,
    most GUI backends have ``window`` and ``toolbar`` attributes that give
    access to the native GUI widgets of the respective framework.

    Attributes
    ----------
    canvas : `FigureCanvasBase`
        The backend-specific canvas instance.

    num : int or str
        The figure number.

    key_press_handler_id : int
        The default key handler cid, when using the toolmanager.
        To disable the default key press handling use::

            figure.canvas.mpl_disconnect(
                figure.canvas.manager.key_press_handler_id)

    button_press_handler_id : int
        The default mouse button handler cid, when using the toolmanager.
        To disable the default button press handling use::

            figure.canvas.mpl_disconnect(
                figure.canvas.manager.button_press_handler_id)
    """
    _toolbar2_class = None
    _toolmanager_toolbar_class = None

    def __init__(self, canvas, num):
        self.canvas = canvas
        canvas.manager = self
        self.num = num
        self.set_window_title(f'Figure {num:d}')
        self.key_press_handler_id = None
        self.button_press_handler_id = None
        if rcParams['toolbar'] != 'toolmanager':
            self.key_press_handler_id = self.canvas.mpl_connect('key_press_event', key_press_handler)
            self.button_press_handler_id = self.canvas.mpl_connect('button_press_event', button_press_handler)
        self.toolmanager = ToolManager(canvas.figure) if mpl.rcParams['toolbar'] == 'toolmanager' else None
        if mpl.rcParams['toolbar'] == 'toolbar2' and self._toolbar2_class:
            self.toolbar = self._toolbar2_class(self.canvas)
        elif mpl.rcParams['toolbar'] == 'toolmanager' and self._toolmanager_toolbar_class:
            self.toolbar = self._toolmanager_toolbar_class(self.toolmanager)
        else:
            self.toolbar = None
        if self.toolmanager:
            tools.add_tools_to_manager(self.toolmanager)
            if self.toolbar:
                tools.add_tools_to_container(self.toolbar)

        @self.canvas.figure.add_axobserver
        def notify_axes_change(fig):
            if self.toolmanager is None and self.toolbar is not None:
                self.toolbar.update()

    @classmethod
    def create_with_canvas(cls, canvas_class, figure, num):
        """
        Create a manager for a given *figure* using a specific *canvas_class*.

        Backends should override this method if they have specific needs for
        setting up the canvas or the manager.
        """
        return cls(canvas_class(figure), num)

    @classmethod
    def start_main_loop(cls):
        """
        Start the main event loop.

        This method is called by `.FigureManagerBase.pyplot_show`, which is the
        implementation of `.pyplot.show`.  To customize the behavior of
        `.pyplot.show`, interactive backends should usually override
        `~.FigureManagerBase.start_main_loop`; if more customized logic is
        necessary, `~.FigureManagerBase.pyplot_show` can also be overridden.
        """

    @classmethod
    def pyplot_show(cls, *, block=None):
        """
        Show all figures.  This method is the implementation of `.pyplot.show`.

        To customize the behavior of `.pyplot.show`, interactive backends
        should usually override `~.FigureManagerBase.start_main_loop`; if more
        customized logic is necessary, `~.FigureManagerBase.pyplot_show` can
        also be overridden.

        Parameters
        ----------
        block : bool, optional
            Whether to block by calling ``start_main_loop``.  The default,
            None, means to block if we are neither in IPython's ``%pylab`` mode
            nor in ``interactive`` mode.
        """
        managers = Gcf.get_all_fig_managers()
        if not managers:
            return
        for manager in managers:
            try:
                manager.show()
            except NonGuiException as exc:
                _api.warn_external(str(exc))
        if block is None:
            pyplot_show = getattr(sys.modules.get('matplotlib.pyplot'), 'show', None)
            ipython_pylab = hasattr(pyplot_show, '_needmain')
            block = not ipython_pylab and (not is_interactive())
        if block:
            cls.start_main_loop()

    def show(self):
        """
        For GUI backends, show the figure window and redraw.
        For non-GUI backends, raise an exception, unless running headless (i.e.
        on Linux with an unset DISPLAY); this exception is converted to a
        warning in `.Figure.show`.
        """
        if sys.platform == 'linux' and (not os.environ.get('DISPLAY')):
            return
        raise NonGuiException(f'{type(self.canvas).__name__} is non-interactive, and thus cannot be shown')

    def destroy(self):
        pass

    def full_screen_toggle(self):
        pass

    def resize(self, w, h):
        """For GUI backends, resize the window (in physical pixels)."""

    def get_window_title(self):
        """
        Return the title text of the window containing the figure, or None
        if there is no window (e.g., a PS backend).
        """
        return 'image'

    def set_window_title(self, title):
        """
        Set the title text of the window containing the figure.

        This has no effect for non-GUI (e.g., PS) backends.

        Examples
        --------
        >>> fig = plt.figure()
        >>> fig.canvas.manager.set_window_title('My figure')
        """