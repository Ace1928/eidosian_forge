import contextlib
import os
import signal
import socket
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib._pylab_helpers import Gcf
from . import _macosx
from .backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import (
class FigureManagerMac(_macosx.FigureManager, FigureManagerBase):
    _toolbar2_class = NavigationToolbar2Mac

    def __init__(self, canvas, num):
        self._shown = False
        _macosx.FigureManager.__init__(self, canvas)
        icon_path = str(cbook._get_data_path('images/matplotlib.pdf'))
        _macosx.FigureManager.set_icon(icon_path)
        FigureManagerBase.__init__(self, canvas, num)
        self._set_window_mode(mpl.rcParams['macosx.window_mode'])
        if self.toolbar is not None:
            self.toolbar.update()
        if mpl.is_interactive():
            self.show()
            self.canvas.draw_idle()

    def _close_button_pressed(self):
        Gcf.destroy(self)
        self.canvas.flush_events()

    def destroy(self):
        while self.canvas._timers:
            timer = self.canvas._timers.pop()
            timer.stop()
        super().destroy()

    @classmethod
    def start_main_loop(cls):
        with _maybe_allow_interrupt():
            _macosx.show()

    def show(self):
        if not self._shown:
            self._show()
            self._shown = True
        if mpl.rcParams['figure.raise_window']:
            self._raise()