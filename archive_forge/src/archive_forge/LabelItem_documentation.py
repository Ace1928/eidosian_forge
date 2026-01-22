from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtWidgets, QtGui
from .GraphicsWidget import GraphicsWidget
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor
Set the text and text properties in the label. Accepts optional arguments for auto-generating
        a CSS style string:

        ==================== ==============================
        **Style Arguments:**
        family               (str) example: 'Cantarell'
        color                (str) example: '#CCFF00'
        size                 (str) example: '8pt'
        bold                 (bool)
        italic               (bool)
        ==================== ==============================
        