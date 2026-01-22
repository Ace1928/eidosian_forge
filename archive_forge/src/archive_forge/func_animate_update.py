import numpy as np
import holoviews as hv
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import Slider, Button
def animate_update():
    year = slider.value + 1
    if year > end:
        year = start
    slider.value = year