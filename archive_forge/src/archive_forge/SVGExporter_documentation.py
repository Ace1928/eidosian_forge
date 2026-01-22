import contextlib
import re
import xml.dom.minidom as xml
import numpy as np
from .. import debug
from .. import functions as fn
from ..parametertree import Parameter
from ..Qt import QtCore, QtGui, QtSvg, QtWidgets
from .Exporter import Exporter
This function is intended to work around some issues with Qt's SVG generator
    and SVG in general.

    .. warning::
        This function, while documented, is not considered part of the public
        API. The reason for its documentation is for ease of referencing by
        :func:`~pyqtgraph.GraphicsItem.generateSvg`. There should be no need
        to call this function explicitly.

    1. Qt SVG does not implement clipping paths. This is absurd.
    The solution is to let Qt generate SVG for each item independently,
    then glue them together manually with clipping.  The format Qt generates 
    for all items looks like this:
        
    .. code-block:: xml
    
        <g>
            <g transform="matrix(...)">
                one or more of: <path/> or <polyline/> or <text/>
            </g>
            <g transform="matrix(...)">
                one or more of: <path/> or <polyline/> or <text/>
            </g>
            . . .
        </g>
        
    2. There seems to be wide disagreement over whether path strokes
    should be scaled anisotropically.  Given that both inkscape and 
    illustrator seem to prefer isotropic scaling, we will optimize for
    those cases.

    .. note::
        
        see: http://web.mit.edu/jonas/www/anisotropy/

    3. Qt generates paths using non-scaling-stroke from SVG 1.2, but
    inkscape only supports 1.1.

    Both 2 and 3 can be addressed by drawing all items in world coordinates.

    Parameters
    ----------
    item : :class:`~pyqtgraph.GraphicsItem`
        GraphicsItem to generate SVG of
    nodes : dict of str, optional
        dictionary keyed on graphics item names, values contains the 
        XML elements, by default None
    root : :class:`~pyqtgraph.GraphicsItem`, optional
        root GraphicsItem, if none, assigns to `item`, by default None
    options : dict of str, optional
        Options to be applied to the generated XML, by default None

    Returns
    -------
    tuple
        tuple where first element is XML element, second element is 
        a list of child GraphicItems XML elements
    