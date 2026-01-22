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
def _maximal_cusp_area(mfd):
    if not hasattr(mfd, 'cusp_area_matrix'):
        return 5.0
    try:
        mfd = mfd.copy()
        mfd.dehn_fill(mfd.num_cusps() * [(0, 0)])
        mfd.init_hyperbolic_structure(force_recompute=True)
        m = mfd.cusp_area_matrix(method='trigDependent')
        return math.sqrt(max([m[i, i] for i in range(mfd.num_cusps())]))
    except Exception as e:
        print('Exception while trying to compute maximal cusp area:', e)
        return 5.0