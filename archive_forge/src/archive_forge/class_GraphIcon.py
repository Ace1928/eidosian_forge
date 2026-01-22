import os.path as op
import warnings
from ..Qt import QtGui, QtWidgets
class GraphIcon:
    """An icon place holder for lazy loading of QIcons

    The icon must reside in the icons folder and the path refers to the full
    name including suffix of the icon file, e.g.:

        tiny = GraphIcon("tiny.png")

    Icons can be later retrieved via the function `getGraphIcon` and providing
    the name:

        tiny = getGraphIcon("tiny")
    """

    def __init__(self, path):
        self._path = path
        name = path.split('.')[0]
        _ICON_REGISTRY[name] = self
        self._icon = None

    def _build_qicon(self):
        icon = QtGui.QIcon(op.join(op.dirname(__file__), self._path))
        name = self._path.split('.')[0]
        _ICON_REGISTRY[name] = icon
        self._icon = icon

    @property
    def qicon(self):
        if self._icon is None:
            self._build_qicon()
        return self._icon