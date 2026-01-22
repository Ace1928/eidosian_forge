import itertools
import math
import weakref
from collections import OrderedDict
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
class SymbolAtlas(object):
    """
    Used to efficiently construct a single QPixmap containing all rendered symbols
    for a ScatterPlotItem. This is required for fragment rendering.

    Use example:
        atlas = SymbolAtlas()
        sc1 = atlas[[('o', 5, QPen(..), QBrush(..))]]
        sc2 = atlas[[('t', 10, QPen(..), QBrush(..))]]
        pm = atlas.pixmap

    """
    _idGenerator = itertools.count()

    def __init__(self):
        self._dpr = 1.0
        self.clear()

    def __getitem__(self, styles):
        """
        Given a list of tuples, (symbol, size, pen, brush), return a list of coordinates of
        corresponding symbols within the atlas. Note that these coordinates may change if the atlas is rebuilt.
        """
        keys = self._keys(styles)
        new = {key: style for key, style in zip(keys, styles) if key not in self._coords}
        if new:
            self._extend(new)
        return list(map(self._coords.__getitem__, keys))

    def __len__(self):
        return len(self._coords)

    def devicePixelRatio(self):
        return self._dpr

    def setDevicePixelRatio(self, dpr):
        self._dpr = dpr

    @property
    def pixmap(self):
        if self._pixmap is None:
            self._pixmap = self._createPixmap()
        return self._pixmap

    @property
    def maxWidth(self):
        return self._maxWidth / self._dpr

    def rebuild(self, styles=None):
        profiler = debug.Profiler()
        if styles is None:
            data = []
        else:
            keys = set(self._keys(styles))
            data = list(self._itemData(keys))
        self.clear()
        if data:
            self._extendFromData(data)

    def clear(self):
        self._data = np.zeros((0, 0, 4), dtype=np.ubyte)
        self._coords = {}
        self._pixmap = None
        self._maxWidth = 0
        self._totalWidth = 0
        self._totalArea = 0
        self._pos = (0, 0)
        self._rowShape = (0, 0)

    def diagnostics(self):
        n = len(self)
        w, h, _ = self._data.shape
        a = self._totalArea
        return dict(count=n, width=w, height=h, area=w * h, area_used=1.0 if n == 0 else a / (w * h), squareness=1.0 if n == 0 else 2 * w * h / (w ** 2 + h ** 2))

    def _keys(self, styles):

        def getId(obj):
            try:
                return obj._id
            except AttributeError:
                obj._id = next(SymbolAtlas._idGenerator)
                return obj._id
        return [(symbol if isinstance(symbol, (str, int)) else getId(symbol), size, getId(pen), getId(brush)) for symbol, size, pen, brush in styles]

    def _itemData(self, keys):
        for key in keys:
            y, x, h, w = self._coords[key]
            yield (key, self._data[x:x + w, y:y + h])

    def _extend(self, styles):
        profiler = debug.Profiler()
        images = []
        data = []
        for key, style in styles.items():
            img = renderSymbol(*style, dpr=self._dpr)
            arr = fn.ndarray_from_qimage(img)
            images.append(img)
            data.append((key, arr))
        profiler('render')
        self._extendFromData(data)
        profiler('insert')

    def _extendFromData(self, data):
        self._pack(data)
        wNew, hNew = self._minDataShape()
        wOld, hOld, _ = self._data.shape
        if wNew > wOld or hNew > hOld:
            arr = np.zeros((wNew, hNew, 4), dtype=np.ubyte)
            arr[:wOld, :hOld] = self._data
            self._data = arr
        for key, arr in data:
            y, x, h, w = self._coords[key]
            self._data[x:x + w, y:y + h] = arr
        self._pixmap = None

    def _pack(self, data):
        n = len(self)
        wMax = self._maxWidth
        wSum = self._totalWidth
        aSum = self._totalArea
        x, y = self._pos
        wRow, hRow = self._rowShape
        for _, arr in data:
            w, h, _ = arr.shape
            wMax = max(w, wMax)
            wSum += w
            aSum += w * h
        n += len(data)
        wRowEst = int(wSum / n ** 0.5)
        if wRowEst > 2 * wRow:
            wRow = wRowEst
        wRow = max(wMax, wRow)
        for key, arr in sorted(data, key=lambda data: data[1].shape[1]):
            w, h, _ = arr.shape
            if x + w > wRow:
                x = 0
                y += hRow
                hRow = h
            hRow = max(h, hRow)
            self._coords[key] = (y, x, h, w)
            x += w
        self._maxWidth = wMax
        self._totalWidth = wSum
        self._totalArea = aSum
        self._pos = (x, y)
        self._rowShape = (wRow, hRow)

    def _minDataShape(self):
        x, y = self._pos
        w, h = self._rowShape
        return (int(w), int(y + h))

    def _createPixmap(self):
        profiler = debug.Profiler()
        if self._data.size == 0:
            pm = QtGui.QPixmap(0, 0)
        else:
            img = fn.ndarray_to_qimage(self._data, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
            pm = QtGui.QPixmap(img)
        return pm