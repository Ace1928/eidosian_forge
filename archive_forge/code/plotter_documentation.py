from __future__ import annotations
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pymatgen.util.plotting import pretty_plot
Save the plot to an image file.

        Args:
            filename (str): Filename to save to. Must include extension to specify image format.
            width: Width of the plot. Defaults to 8 in.
            height: Height of the plot. Defaults to 6 in.
        