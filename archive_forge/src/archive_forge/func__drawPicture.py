import numpy as np
from .. import Qt, colormap
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def _drawPicture(self) -> QtGui.QPicture:
    picture = QtGui.QPicture()
    painter = QtGui.QPainter(picture)
    if self.edgecolors is None:
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
    else:
        painter.setPen(self.edgecolors)
        if self.antialiasing:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    lut = self.lut_qcolor
    scale = len(lut) - 1
    lo, hi = (self.levels[0], self.levels[1])
    rng = hi - lo
    if rng == 0:
        rng = 1
    norm = fn.rescaleData(self.z, scale / rng, lo, dtype=int, clip=(0, len(lut) - 1))
    if Qt.QT_LIB.startswith('PyQt'):
        drawConvexPolygon = lambda x: painter.drawConvexPolygon(*x)
    else:
        drawConvexPolygon = painter.drawConvexPolygon
    self.quads.resize(self.z.shape[0], self.z.shape[1])
    memory = self.quads.ndarray()
    memory[..., 0] = self.x.ravel()
    memory[..., 1] = self.y.ravel()
    polys = self.quads.instances()
    color_indices, counts = np.unique(norm, return_counts=True)
    sorted_indices = np.argsort(norm, axis=None)
    offset = 0
    for coloridx, cnt in zip(color_indices, counts):
        indices = sorted_indices[offset:offset + cnt]
        offset += cnt
        painter.setBrush(lut[coloridx])
        for idx in indices:
            drawConvexPolygon(polys[idx])
    painter.end()
    return picture