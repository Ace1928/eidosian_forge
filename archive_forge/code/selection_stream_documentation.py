import numpy as np
import holoviews as hv
from holoviews.streams import Selection1D

An example app demonstrating how to use the HoloViews API to generate
a bokeh app with complex interactivity. Uses a Selection1D stream
to compute the mean y-value of the current selection.

The app can be served using:

    bokeh serve --show selection_stream.py
