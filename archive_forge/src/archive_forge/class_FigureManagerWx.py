import functools
import logging
import math
import pathlib
import sys
import weakref
import numpy as np
import PIL.Image
import matplotlib as mpl
from matplotlib.backend_bases import (
from matplotlib import _api, cbook, backend_tools
from matplotlib._pylab_helpers import Gcf
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import wx
class FigureManagerWx(FigureManagerBase):
    """
    Container/controller for the FigureCanvas and GUI frame.

    It is instantiated by Gcf whenever a new figure is created.  Gcf is
    responsible for managing multiple instances of FigureManagerWx.

    Attributes
    ----------
    canvas : `FigureCanvas`
        a FigureCanvasWx(wx.Panel) instance
    window : wxFrame
        a wxFrame instance - wxpython.org/Phoenix/docs/html/Frame.html
    """

    def __init__(self, canvas, num, frame):
        _log.debug('%s - __init__()', type(self))
        self.frame = self.window = frame
        super().__init__(canvas, num)

    @classmethod
    def create_with_canvas(cls, canvas_class, figure, num):
        wxapp = wx.GetApp() or _create_wxapp()
        frame = FigureFrameWx(num, figure, canvas_class=canvas_class)
        manager = figure.canvas.manager
        if mpl.is_interactive():
            manager.frame.Show()
            figure.canvas.draw_idle()
        return manager

    @classmethod
    def start_main_loop(cls):
        if not wx.App.IsMainLoopRunning():
            wxapp = wx.GetApp()
            if wxapp is not None:
                wxapp.MainLoop()

    def show(self):
        self.frame.Show()
        self.canvas.draw()
        if mpl.rcParams['figure.raise_window']:
            self.frame.Raise()

    def destroy(self, *args):
        _log.debug('%s - destroy()', type(self))
        frame = self.frame
        if frame:
            wx.CallAfter(frame.Close)

    def full_screen_toggle(self):
        self.frame.ShowFullScreen(not self.frame.IsFullScreen())

    def get_window_title(self):
        return self.window.GetTitle()

    def set_window_title(self, title):
        self.window.SetTitle(title)

    def resize(self, width, height):
        self.window.SetSize(self.window.ClientToWindowSize(wx.Size(math.ceil(width), math.ceil(height))))