from colorsys import hsv_to_rgb, hls_to_rgb
from .libmp import NoConvergence
from .libmp.backend import xrange
class VisualizationMethods(object):
    plot_ignore = (ValueError, ArithmeticError, ZeroDivisionError, NoConvergence)