from time import perf_counter
import pyglet.gl as pgl
from sympy.plotting.pygletplot.managed_window import ManagedWindow
from sympy.plotting.pygletplot.plot_camera import PlotCamera
from sympy.plotting.pygletplot.plot_controller import PlotController

        Named Arguments
        ===============

        antialiasing = True
            True OR False
        ortho = False
            True OR False
        invert_mouse_zoom = False
            True OR False
        