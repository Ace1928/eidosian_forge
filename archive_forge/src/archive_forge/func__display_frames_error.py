import ipywidgets as widgets
from traitlets import List, Unicode, Dict, observe, Integer
from .basedatatypes import BaseFigure, BasePlotlyType
from .callbacks import BoxSelector, LassoSelector, InputDeviceState, Points
from .serializers import custom_serializers
from .version import __frontend_version__
@staticmethod
def _display_frames_error():
    """
        Display an informative error when user attempts to set frames on a
        FigureWidget

        Raises
        ------
        ValueError
            always
        """
    msg = '\nFrames are not supported by the plotly.graph_objs.FigureWidget class.\nNote: Frames are supported by the plotly.graph_objs.Figure class'
    raise ValueError(msg)