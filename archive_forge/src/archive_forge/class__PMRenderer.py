from reportlab.graphics.shapes import *
from reportlab.graphics.renderbase import getStateDelta, renderScaledDrawing
from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import isUnicode
from reportlab import rl_config
from .utils import setFont as _setFont, RenderPMError
import os, sys
from io import BytesIO, StringIO
from math import sin, cos, pi, ceil
from reportlab.graphics.renderbase import Renderer
class _PMRenderer(Renderer):
    """This draws onto a pix map image. It needs to be a class
    rather than a function, as some image-specific state tracking is
    needed outside of the state info in the SVG model."""

    def pop(self):
        self._tracker.pop()
        self.applyState()

    def push(self, node):
        deltas = getStateDelta(node)
        self._tracker.push(deltas)
        self.applyState()

    def applyState(self):
        s = self._tracker.getState()
        self._canvas.ctm = s['ctm']
        self._canvas.strokeWidth = s['strokeWidth']
        alpha = s['strokeOpacity']
        if alpha is not None:
            self._canvas.strokeOpacity = alpha
        self._canvas.setStrokeColor(s['strokeColor'])
        self._canvas.lineCap = s['strokeLineCap']
        self._canvas.lineJoin = s['strokeLineJoin']
        self._canvas.fillMode = s['fillMode']
        da = s['strokeDashArray']
        if not da:
            da = None
        else:
            if not isinstance(da, (list, tuple)):
                da = (da,)
            if len(da) != 2 or not isinstance(da[1], (list, tuple)):
                da = (0, da)
        self._canvas.dashArray = da
        alpha = s['fillOpacity']
        if alpha is not None:
            self._canvas.fillOpacity = alpha
        self._canvas.setFillColor(s['fillColor'])
        self._canvas.setFont(s['fontName'], s['fontSize'])

    def initState(self, x, y):
        deltas = self._tracker._combined[-1]
        deltas['transform'] = deltas['ctm'] = self._canvas._baseCTM[0:4] + (x, y)
        self._tracker.push(deltas)
        self.applyState()

    def drawNode(self, node):
        """This is the recursive method called for each node
        in the tree"""
        self.push(node)
        self.drawNodeDispatcher(node)
        self.pop()

    def drawRect(self, rect):
        c = self._canvas
        if rect.rx == rect.ry == 0:
            c.rect(rect.x, rect.y, rect.width, rect.height)
        else:
            c.roundRect(rect.x, rect.y, rect.width, rect.height, rect.rx, rect.ry)

    def drawLine(self, line):
        self._canvas.line(line.x1, line.y1, line.x2, line.y2)

    def drawImage(self, image):
        path = image.path
        if isinstance(path, str):
            if not (path and os.path.isfile(path)):
                return
            im = _getImage().open(path).convert('RGB')
        elif hasattr(path, 'convert'):
            im = path.convert('RGB')
        else:
            return
        srcW, srcH = im.size
        dstW, dstH = (image.width, image.height)
        if dstW is None:
            dstW = srcW
        if dstH is None:
            dstH = srcH
        self._canvas._aapixbuf(image.x, image.y, dstW, dstH, im if self._canvas._backend == 'rlPyCairo' else (im.tobytes if hasattr(im, 'tobytes') else im.tostring)(), srcW, srcH, 3)

    def drawCircle(self, circle):
        c = self._canvas
        c.circle(circle.cx, circle.cy, circle.r)
        c.fillstrokepath()

    def drawPolyLine(self, polyline, _doClose=0):
        P = polyline.points
        assert len(P) >= 2, 'Polyline must have 1 or more points'
        c = self._canvas
        c.pathBegin()
        c.moveTo(P[0], P[1])
        for i in range(2, len(P), 2):
            c.lineTo(P[i], P[i + 1])
        if _doClose:
            c.pathClose()
            c.pathFill()
        c.pathStroke()

    def drawEllipse(self, ellipse):
        c = self._canvas
        c.ellipse(ellipse.cx, ellipse.cy, ellipse.rx, ellipse.ry)
        c.fillstrokepath()

    def drawPolygon(self, polygon):
        self.drawPolyLine(polygon, _doClose=1)

    def drawString(self, stringObj):
        canv = self._canvas
        fill = canv.fillColor
        textRenderMode = getattr(stringObj, 'textRenderMode', 0)
        if fill is not None or textRenderMode:
            S = self._tracker.getState()
            text_anchor = S['textAnchor']
            fontName = S['fontName']
            fontSize = S['fontSize']
            text = stringObj.text
            x = stringObj.x
            y = stringObj.y
            if not text_anchor in ['start', 'inherited']:
                textLen = stringWidth(text, fontName, fontSize)
                if text_anchor == 'end':
                    x -= textLen
                elif text_anchor == 'middle':
                    x -= textLen / 2
                elif text_anchor == 'numeric':
                    x -= numericXShift(text_anchor, text, textLen, fontName, fontSize, stringObj.encoding)
                else:
                    raise ValueError('bad value for textAnchor ' + str(text_anchor))
            oldTextRenderMode = canv.textRenderMode
            canv.textRenderMode = textRenderMode
            try:
                canv.drawString(x, y, text, _fontInfo=(fontName, fontSize))
            finally:
                canv.textRenderMode = oldTextRenderMode

    def drawPath(self, path):
        c = self._canvas
        if path is EmptyClipPath:
            del c._clipPaths[-1]
            if c._clipPaths:
                P = c._clipPaths[-1]
                icp = P.isClipPath
                P.isClipPath = 1
                self.drawPath(P)
                P.isClipPath = icp
            else:
                c.clipPathClear()
            return
        from reportlab.graphics.shapes import _renderPath
        drawFuncs = (c.moveTo, c.lineTo, c.curveTo, c.pathClose)
        autoclose = getattr(path, 'autoclose', '')

        def rP(forceClose=False):
            c.pathBegin()
            return _renderPath(path, drawFuncs, forceClose=forceClose)
        if path.isClipPath:
            rP()
            c.clipPathSet()
            c._clipPaths.append(path)
        fill = c.fillColor is not None
        stroke = c.strokeColor is not None
        fillMode = getattr(path, 'fillMode', -1)
        if autoclose == 'svg':
            if fill and stroke:
                rP(forceClose=True)
                c.pathFill(fillMode)
                rP()
                c.pathStroke()
            elif fill:
                rP(forceClose=True)
                c.pathFill(fillMode)
            elif stroke:
                rP()
                c.pathStroke()
        elif autoclose == 'pdf':
            rP(forceClose=True)
            if fill:
                c.pathFill(fillMode)
            if stroke:
                c.pathStroke()
        else:
            if rP():
                c.pathFill(fillMode)
            c.pathStroke()