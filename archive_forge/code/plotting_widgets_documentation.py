from ipywidgets import Layout
from traitlets import List, Enum, Int, Bool
from traittypes import DataFrame
from bqplot import Figure, LinearScale, Lines, Label
from bqplot.marks import CATEGORY10
import numpy as np

    Radar chart created from a pandas Dataframe. Each column of the df will be
    represented as a loop in the radar chart. Each row of the df will be
    represented as a spoke of the radar chart

    Attributes
    ----------
    data: DataFrame
        data for the radar
    band_type: {"circle", "polygon"} (default: "circle")
        type of bands to display in the radar
    num_bands: Int (default: 5)
        number of bands on the radar. As of now, this attribute is not
        dynamic and it has to set in the constructor
    data_range: List (default: [0, 1])
        range of data
    fill: Bool(default: True)
        flag which lets us fill the radar loops or not
    