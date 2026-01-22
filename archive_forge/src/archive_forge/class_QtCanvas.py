import copy
from math import *
from qt import *
from qtcanvas import *
from rdkit.sping import pid
class QtCanvas(pid.Canvas):

    def __init__(self, destCanvas, size=(300, 300), name='QtCanvas'):
        self.size = size
        pid.Canvas.__init__(self, size, name)
        self._canvas = destCanvas
        self._brush = QBrush()
        self._pen = QPen()
        self._font = QApplication.font()
        self.objs = []
        self._initOutput()
        self.nObjs = 0

    def _initOutput(self):
        for obj in self.objs:
            if type(obj) == tuple:
                obj[0].hide()
            else:
                obj.hide()
        self.objs = []
        self.nObjs = 0

    def _adjustFont(self, font):
        if font.face:
            self._font.setFamily(font.face)
        self._font.setBold(font.bold)
        self._font.setItalic(font.italic)
        self._font.setPointSize(font.size)
        self._font.setUnderline(font.underline)

    def clear(self):
        self._initOutput()

    def flush(self):
        self._canvas.update()

    def save(self, file=None, format=None):
        self._canvas.update()

    def drawLine(self, x1, y1, x2, y2, color=None, width=None, dash=None, **kwargs):
        """Draw a straight line between x1,y1 and x2,y2."""
        if color:
            if color == pid.transparent:
                return
        elif self.defaultLineColor == pid.transparent:
            return
        else:
            color = self.defaultLineColor
        qColor = _ColorToQt(color)
        if width:
            w = width
        else:
            w = self.defaultLineWidth
        self._pen.setColor(qColor)
        self._pen.setWidth(int(w))
        if dash is not None:
            self._pen.setStyle(Qt.DashLine)
        else:
            self._pen.setStyle(Qt.SolidLine)
        l = QCanvasLine(self._canvas)
        l.setPen(self._pen)
        l.setPoints(x1, y1, x2, y2)
        l.setVisible(1)
        l.setZ(self.nObjs)
        if dash is not None:
            self._pen.setStyle(Qt.SolidLine)
        self.nObjs += 1
        self.objs.append(l)

    def drawPolygon(self, pointlist, edgeColor=None, edgeWidth=None, fillColor=pid.transparent, closed=0, dash=None, **kwargs):
        """drawPolygon(pointlist) -- draws a polygon
    pointlist: a list of (x,y) tuples defining vertices

    """
        pts = []
        for point in pointlist:
            pts += list(point)
        ptArr = QPointArray()
        ptArr.setPoints(pts)
        filling = 0
        if fillColor:
            if fillColor != pid.transparent:
                filling = 1
                qColor = _ColorToQt(fillColor)
                self._brush.setColor(qColor)
        if filling:
            self._brush.setStyle(Qt.SolidPattern)
        else:
            self._brush.setStyle(Qt.NoBrush)
        if not edgeColor:
            edgeColor = self.defaultLineColor
        qColor = _ColorToQt(edgeColor)
        if qColor:
            self._pen.setColor(qColor)
        if edgeWidth is None:
            edgeWidth = self.defaultLineWidth
        self._pen.setWidth(edgeWidth)
        self._pen.setJoinStyle(Qt.RoundJoin)
        if dash is not None:
            self._pen.setStyle(Qt.DashLine)
        else:
            self._pen.setStyle(Qt.SolidLine)
        poly = QCanvasPolygon(self._canvas)
        poly.setPen(self._pen)
        poly.setBrush(self._brush)
        poly.setPoints(ptArr)
        poly.setVisible(1)
        poly.setZ(self.nObjs)
        self.nObjs += 1
        self.objs.append(poly)
        if edgeColor != pid.transparent:
            for i in range(len(pointlist) - 1):
                l = QCanvasLine(self._canvas)
                l.setPoints(pointlist[i][0], pointlist[i][1], pointlist[i + 1][0], pointlist[i + 1][1])
                l.setPen(self._pen)
                l.setVisible(1)
                l.setZ(self.nObjs)
                self.objs.append(l)
            if closed:
                l = QCanvasLine(self._canvas)
                l.setPoints(pointlist[0][0], pointlist[0][1], pointlist[-1][0], pointlist[-1][1])
                l.setPen(self._pen)
                l.setVisible(1)
                l.setZ(self.nObjs)
                self.objs.append(l)
        if dash is not None:
            self._pen.setStyle(Qt.SolidLine)
        self.nObjs += 1

    def drawString(self, s, x, y, font=None, color=None, angle=0, **kwargs):
        if color:
            if color == pid.transparent:
                return
        elif self.defaultLineColor == pid.transparent:
            return
        else:
            color = self.defaultLineColor
        if font is None:
            font = self.defaultFont
        qColor = _ColorToQt(color)
        if font is not None:
            self._adjustFont(font)
        if angle != 0:
            txt = QCanvasRotText(s, self._canvas, angle=angle)
        else:
            txt = QCanvasText(s, self._canvas)
        txt.setTextFlags(Qt.AlignLeft | Qt.AlignVCenter)
        if self._font:
            txt.setFont(self._font)
        txt.setColor(qColor)
        txt.setVisible(1)
        txt.setX(x)
        y -= font.size
        txt.setY(y)
        txt.setZ(self.nObjs)
        self.nObjs += 1
        self.objs.append(txt)

    def drawImage(self, image, x1, y1, x2=None, y2=None, **kwargs):
        """
    """
        from io import StringIO
        sio = StringIO()
        image.save(sio, format='png')
        base = QPixmap()
        base.loadFromData(sio.getvalue())
        pm = QCanvasPixmap(base, QPoint(0, 0))
        pma = QCanvasPixmapArray()
        pma.setImage(0, pm)
        img = QCanvasSprite(pma, self._canvas)
        img.setVisible(1)
        img.setX(x1)
        img.setY(y1)
        self.objs.append((img, base, pm, pma))

    def stringWidth(self, s, font=None):
        """Return the logical width of the string if it were drawn     in the current font (defaults to self.font)."""
        if not font:
            font = self.defaultFont
        if font:
            self._adjustFont(font)
        t = QCanvasText(s, self._canvas)
        t.setFont(self._font)
        rect = t.boundingRect()
        return rect.width()

    def fontAscent(self, font=None):
        if not font:
            font = self.defaultFont
        if font:
            self._adjustFont(font)
        t = QCanvasText('B', self._canvas)
        t.setFont(self._font)
        rect = t.boundingRect()
        return 1.0 * rect.height()

    def fontDescent(self, font=None):
        if not font:
            font = self.defaultFont
        if font:
            self._adjustFont(font)
        t = QCanvasText('B', self._canvas)
        t.setFont(self._font)
        rect1 = t.boundingRect()
        t = QCanvasText('y', self._canvas)
        t.setFont(self._font)
        rect2 = t.boundingRect()
        return 1.0 * (rect2.height() - rect1.height())