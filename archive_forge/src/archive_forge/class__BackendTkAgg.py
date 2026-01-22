from . import _backend_tk
from .backend_agg import FigureCanvasAgg
from ._backend_tk import _BackendTk, FigureCanvasTk
from ._backend_tk import (  # noqa: F401 # pylint: disable=W0611
@_BackendTk.export
class _BackendTkAgg(_BackendTk):
    FigureCanvas = FigureCanvasTkAgg