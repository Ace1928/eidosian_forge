import tkinter
import math
import sys
import time
from tkinter import ttk
from . import gui_utilities
from .gui_utilities import UniformDictController, FpsLabelUpdater
from .view_scale_controller import ViewScaleController
from .raytracing_view import *
from .geodesics_window import GeodesicsWindow
from .hyperboloid_utilities import unit_3_vector_and_distance_to_O13_hyperbolic_translation
from .zoom_slider import Slider, ZoomSlider
def _mouse_gestures_text():
    if sys.platform == 'darwin':
        return u'Move: Click & Drag     Rotate: Shift-Click & Drag     Orbit: ⌘-Click & Drag'
    else:
        return 'Move: Click & Drag     Rotate: Shift-Click & Drag     Orbit: Alt-Click & Drag'