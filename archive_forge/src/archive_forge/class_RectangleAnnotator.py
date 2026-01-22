import sys
from inspect import getmro
import param
from panel.layout import Row, Tabs
from panel.pane import PaneBase
from panel.util import param_name
from .core import DynamicMap, Element, HoloMap, Layout, Overlay, Store, ViewableElement
from .core.util import isscalar
from .element import Curve, Path, Points, Polygons, Rectangles, Table
from .plotting.links import (
from .streams import BoxEdit, CurveEdit, PointDraw, PolyDraw, PolyEdit, Selection1D
class RectangleAnnotator(_GeomAnnotator):
    """
    Annotator which allows drawing and editing Rectangles and associating
    values with each point using a table.
    """
    object = param.ClassSelector(class_=Rectangles, doc='\n        Points element to edit and annotate.')
    _stream_type = BoxEdit
    _link_type = RectanglesTableLink