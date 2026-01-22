from contextlib import nullcontext
from .backend_cairo import FigureCanvasCairo
from .backend_gtk4 import Gtk, FigureCanvasGTK4, _BackendGTK4
@_BackendGTK4.export
class _BackendGTK4Cairo(_BackendGTK4):
    FigureCanvas = FigureCanvasGTK4Cairo