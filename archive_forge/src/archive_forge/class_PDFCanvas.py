import os
from math import cos, pi, sin
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
from . import pdfgen, pdfgeom, pdfmetrics
class PDFCanvas(Canvas):
    """This works by accumulating a list of strings containing
      PDF page marking operators, as you call its methods.  We could
      use a big string but this is more efficient - only concatenate
      it once, with control over line ends.  When
      done, it hands off the stream to a PDFPage object."""

    def __init__(self, size=None, name='pidPDF.pdf', pagesize=DEFAULT_PAGE_SIZE):
        root, ext = os.path.splitext(name)
        if ext == '':
            name = root + '.pdf'
        self.pdf = pdfgen.Canvas(name, pagesize=pagesize, bottomup=0)
        self.pdf.setPageCompression(0)
        self.pdf.setLineCap(2)
        Canvas.__init__(self, size=size, name=name)
        self.pagesize = pagesize
        self.filename = name
        if size is None:
            self.drawingsize = self.pagesize
        else:
            self.drawingsize = size
        self.pageTransitionString = ''
        self.pageNumber = 1
        if self.pagesize != self.drawingsize:
            dx = 0.5 * (self.pagesize[0] - self.drawingsize[0])
            dy = 0.5 * (self.pagesize[1] - self.drawingsize[1])
            self.pdf.translate(dx, dy)

    def _resetDefaults(self):
        """Only used in setup - persist from page to page"""
        self.defaultLineColor = black
        self.defaultFillColor = transparent
        self.defaultLineWidth = 1
        self.defaultFont = Font()
        self.pdf.setLineCap(2)

    def showPage(self):
        """ensure basic settings are the same after a page break"""
        self.pdf.showPage()
        self.defaultFont = self.defaultFont
        self.defaultLineColor = self.defaultLineColor
        self.defaultFillColor = self.defaultFillColor
        self.defaultLineWidth = self.defaultLineWidth
        self.pdf.setLineCap(2)

    def isInteractive(self):
        return 0

    def canUpdate(self):
        return 0

    def clear(self):
        """Not wll defined for file formats, use same as ShowPage"""
        self.showPage()

    def flush(self):
        pass

    def save(self, file=None, format=None):
        """Saves the file.  If holding data, do
            a showPage() to save them having to."""
        if self.pdf.pageHasData():
            self.pdf.showPage()
        if hasattr(file, 'write'):
            self.pdf.save(fileobj=file)
        elif isinstance(file, str):
            self.pdf.save(filename=file)
        else:
            self.pdf.save()

    def setInfoLine(self, s):
        self.pdf.setTitle(s)

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key == 'defaultLineColor':
            self._updateLineColor(value)
        elif key == 'defaultLineWidth':
            self._updateLineWidth(value)
        elif key == 'defaultFillColor':
            self._updateFillColor(value)
        elif key == 'defaultFont':
            self._updateFont(value)

    def _updateLineColor(self, color):
        """Triggered when someone assigns to defaultLineColor"""
        self.pdf.setStrokeColorRGB(color.red, color.green, color.blue)

    def _updateFillColor(self, color):
        """Triggered when someone assigns to defaultFillColor"""
        self.pdf.setFillColorRGB(color.red, color.green, color.blue)

    def _updateLineWidth(self, width):
        """Triggered when someone assigns to defaultLineWidth"""
        self.pdf.setLineWidth(width)

    def _updateFont(self, font):
        """Triggered when someone assigns to defaultFont"""
        psfont = self._findPostScriptFontName(font)
        self.pdf.setFont(psfont, font.size)

    def _findPostScriptFontName(self, font):
        """Attempts to return proper font name."""
        if not font.face:
            face = 'serif'
        else:
            face = font.face.lower()
        while face in font_face_map:
            face = font_face_map[face]
        psname = ps_font_map[face, font.bold, font.italic]
        return psname

    def _escape(self, s):
        """PDF escapes are like Python ones, but brackets need slashes before them too.
            Use Python's repr function and chop off the quotes first"""
        s = repr(s)[1:-1]
        s = s.replace('(', '\\(')
        s = s.replace(')', '\\)')
        return s

    def resetDefaults(self):
        """If you drop down to a lower level, PIDDLE can lose
            track of the current graphics state.  Calling this after
            wards ensures that the canvas is updated to the same
            defaults as PIDDLE thinks they should be."""
        self.defaultFont = self.defaultFont
        self.defaultLineColor = self.defaultLineColor
        self.defaultFillColor = self.defaultFillColor
        self.defaultLineWidth = self.defaultLineWidth

    def stringWidth(self, s, font=None):
        """Return the logical width of the string if it were drawn             in the current font (defaults to self.font)."""
        if not font:
            font = self.defaultFont
        fontname = self._findPostScriptFontName(font)
        return pdfmetrics.stringwidth(s, fontname) * font.size * 0.001

    def fontHeight(self, font=None):
        if not font:
            font = self.defaultFont
        return font.size

    def fontAscent(self, font=None):
        if not font:
            font = self.defaultFont
        fontname = self._findPostScriptFontName(font)
        return pdfmetrics.ascent_descent[fontname][0] * 0.001 * font.size

    def fontDescent(self, font=None):
        if not font:
            font = self.defaultFont
        fontname = self._findPostScriptFontName(font)
        return -pdfmetrics.ascent_descent[fontname][1] * 0.001 * font.size

    def _endPath(self, path, edgeColor, fillColor):
        """in PIDDLE, the edge and fil colors might be transparent,
            and might also be None, in which case they should be taken
            from the defaults.  This leads to a standard 10 lines of code
            when closing each shape, which are wrapped up here.  Use
            these if you implement new PIDDLE shapes."""
        fill = fillColor or self.defaultFillColor
        edge = edgeColor or self.defaultLineColor
        if fill == transparent and edge == transparent:
            pass
        else:
            self.pdf.drawPath(path, edge != transparent, fill != transparent)

    def drawLine(self, x1, y1, x2, y2, color=None, width=None, dash=None, **kwargs):
        """Calls the underlying methods in pdfgen.canvas.  For the
            highest performance, use canvas.setDefaultFont and
            canvas.setLineWidth, and draw batches of similar
            lines together."""
        if color:
            self._updateLineColor(color)
        if width:
            self._updateLineWidth(width)
        self.pdf.line(x1, y1, x2, y2)
        if color:
            self._updateLineColor(self.defaultLineColor)
        if width:
            self._updateLineWidth(self.defaultLineWidth)

    def drawLines(self, lineList, color=None, width=None, dash=None, **kwargs):
        """Draws several distinct lines, all with same color
            and width, efficiently"""
        if color:
            self._updateLineColor(color)
        if width:
            self._updateLineWidth(width)
        self.pdf.lines(lineList)
        if color:
            self._updateLineColor(self.defaultLineColor)
        if width:
            self._updateLineWidth(self.defaultLineWidth)

    def drawString(self, s, x, y, font=None, color=None, angle=0, **kwargs):
        """As it says, but many options to process.  It translates
            user space rather than text space, in case underlining is
            needed on rotated text.  It cheats and does literals
            for efficiency, avoiding changing the python graphics state."""
        self.pdf.addLiteral('%begin drawString')
        col = color or self.defaultLineColor
        if col != transparent:
            if '\n' in s or '\r' in s:
                s = s.replace('\r\n', '\n')
                s = s.replace('\n\r', '\n')
                lines = s.split('\n')
            else:
                lines = [s]
            fnt = font or self.defaultFont
            self._updateFont(fnt)
            text = self.pdf._escape(s)
            textobj = self.pdf.beginText()
            if col != self.defaultFillColor:
                textobj.setFillColorRGB(col.red, col.green, col.blue)
            if angle != 0:
                co = cos(angle * pi / 180.0)
                si = sin(angle * pi / 180.0)
                textobj.setTextTransform(co, -si, si, co, x, y)
            else:
                textobj.setTextOrigin(x, y)
            for line in lines:
                if fnt.underline:
                    ycursor = textobj.getY()
                    dy = 0.5 * self.fontDescent(fnt)
                    width = self.stringWidth(line, fnt)
                    linewidth = fnt.size * 0.1
                    self.pdf.saveState()
                    self.pdf.setLineWidth(linewidth)
                    self.pdf.translate(x, y)
                    if angle != 0:
                        self.pdf.rotate(-angle)
                    self.pdf.translate(0, ycursor - y)
                    self.pdf.line(0, dy, width, dy)
                    self.pdf.restoreState()
                    lasty = ycursor
                textobj.textLine(line)
            self.pdf.drawText(textobj)
        self.pdf.addLiteral('%end drawString')

    def drawCurve(self, x1, y1, x2, y2, x3, y3, x4, y4, edgeColor=None, edgeWidth=None, fillColor=None, closed=0, dash=None, **kwargs):
        """This could do two totally different things.  If not closed,
            just does a bezier curve so fill is irrelevant.  If closed,
            it is actually a filled shape."""
        if closed:
            if edgeColor:
                self._updateLineColor(edgeColor)
            if edgeWidth:
                self._updateLineWidth(edgeWidth)
            if fillColor:
                self._updateFillColor(fillColor)
            p = self.pdf.beginPath()
            p.moveTo(x1, y1)
            p.curveTo(x2, y2, x3, y3, x4, y4)
            p.close()
            self._endPath(p, edgeColor, fillColor)
            if edgeColor:
                self._updateLineColor(self.defaultLineColor)
            if edgeWidth:
                self._updateLineWidth(self.defaultLineWidth)
            if fillColor:
                self._updateFillColor(self.defaultFillColor)
        else:
            if edgeColor:
                self._updateLineColor(edgeColor)
            if edgeWidth:
                self._updateLineWidth(edgeWidth)
            self.pdf.bezier(x1, y1, x2, y2, x3, y3, x4, y4)
            if edgeColor:
                self._updateLineColor(self.defaultLineColor)
            if edgeWidth:
                self._updateLineWidth(self.defaultLineWidth)

    def drawRect(self, x1, y1, x2, y2, edgeColor=None, edgeWidth=None, fillColor=None, dash=None, **kwargs):
        if edgeColor:
            self._updateLineColor(edgeColor)
        if edgeWidth:
            self._updateLineWidth(edgeWidth)
        if fillColor:
            self._updateFillColor(fillColor)
        p = self.pdf.beginPath()
        p.rect(x1, y1, x2 - x1, y2 - y1)
        self._endPath(p, edgeColor, fillColor)
        if edgeColor:
            self._updateLineColor(self.defaultLineColor)
        if edgeWidth:
            self._updateLineWidth(self.defaultLineWidth)
        if fillColor:
            self._updateFillColor(self.defaultFillColor)

    def drawEllipse(self, x1, y1, x2, y2, edgeColor=None, edgeWidth=None, fillColor=None, dash=None, **kwargs):
        if edgeColor:
            self._updateLineColor(edgeColor)
        if edgeWidth:
            self._updateLineWidth(edgeWidth)
        if fillColor:
            self._updateFillColor(fillColor)
        p = self.pdf.beginPath()
        p.ellipse(x1, y1, x2 - x1, y2 - y1)
        self._endPath(p, edgeColor, fillColor)
        if edgeColor:
            self._updateLineColor(self.defaultLineColor)
        if edgeWidth:
            self._updateLineWidth(self.defaultLineWidth)
        if fillColor:
            self._updateFillColor(self.defaultFillColor)

    def drawArc(self, x1, y1, x2, y2, startAng=0, extent=90, edgeColor=None, edgeWidth=None, fillColor=None, dash=None, **kwargs):
        """This draws a PacMan-type shape connected to the centre.  One
            idiosyncrasy - if you specify an edge color, it apples to the
            outer curved rim but not the radial edges."""
        if edgeColor:
            self._updateLineColor(edgeColor)
        if edgeWidth:
            self._updateLineWidth(edgeWidth)
        if fillColor:
            self._updateFillColor(fillColor)
        pointList = pdfgeom.bezierArc(x1, y1, x2, y2, -startAng, -extent)
        start = pointList[0]
        end = pointList[-1]
        x_cen = 0.5 * (x1 + x2)
        y_cen = 0.5 * (y1 + y2)
        p = self.pdf.beginPath()
        p.moveTo(x_cen, y_cen)
        p.lineTo(start[0], start[1])
        for curve in pointList:
            p.curveTo(curve[2], curve[3], curve[4], curve[5], curve[6], curve[7])
        p.close()
        self._endPath(p, transparent, fillColor)
        p2 = self.pdf.beginPath()
        p2.moveTo(start[0], start[1])
        for curve in pointList:
            p2.curveTo(curve[2], curve[3], curve[4], curve[5], curve[6], curve[7])
        self._endPath(p2, edgeColor, transparent)
        if edgeColor:
            self._updateLineColor(self.defaultLineColor)
        if edgeWidth:
            self._updateLineWidth(self.defaultLineWidth)
        if fillColor:
            self._updateFillColor(self.defaultFillColor)

    def drawPolygon(self, pointlist, edgeColor=None, edgeWidth=None, fillColor=None, closed=0, dash=None, **kwargs):
        """As it says.  Easy with paths!"""
        if edgeColor:
            self._updateLineColor(edgeColor)
        if edgeWidth:
            self._updateLineWidth(edgeWidth)
        if fillColor:
            self._updateFillColor(fillColor)
        p = self.pdf.beginPath()
        p.moveTo(pointlist[0][0], pointlist[0][1])
        for point in pointlist[1:]:
            p.lineTo(point[0], point[1])
        if closed:
            p.close()
        self._endPath(p, edgeColor, fillColor)
        if edgeColor:
            self._updateLineColor(self.defaultLineColor)
        if edgeWidth:
            self._updateLineWidth(self.defaultLineWidth)
        if fillColor:
            self._updateFillColor(self.defaultFillColor)

    def drawImage(self, image, x1, y1, x2=None, y2=None, **kwargs):
        """Draw a PIL Image or image filename into the specified rectangle.
            If x2 and y2 are omitted, they are calculated from the image size.
            """
        im_width, im_height = image.size
        if not x2:
            x2 = x1 + im_width
        if not y2:
            y2 = y1 + im_height
        self.pdf.saveState()
        self.pdf.translate(x1, y1)
        self.pdf.drawInlineImage(image, 0, 0, abs(x1 - x2), abs(y1 - y2))
        self.pdf.restoreState()

    def drawLiteral(self, literal):
        self.code.append(literal)