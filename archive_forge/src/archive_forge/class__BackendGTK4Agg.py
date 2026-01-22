import numpy as np
from .. import cbook
from . import backend_agg, backend_gtk4
from .backend_gtk4 import Gtk, _BackendGTK4
import cairo  # Presence of cairo is already checked by _backend_gtk.
@_BackendGTK4.export
class _BackendGTK4Agg(_BackendGTK4):
    FigureCanvas = FigureCanvasGTK4Agg