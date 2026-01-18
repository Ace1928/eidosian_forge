import uuid
from unittest import TestCase
from unittest.mock import Mock
import plotly.graph_objs as go
from holoviews import Tiles
from holoviews.plotting.plotly.callbacks import (
from holoviews.streams import (

    Build a collection of plots, callbacks, and streams for a given callback class and
    a list of trace_uids
    