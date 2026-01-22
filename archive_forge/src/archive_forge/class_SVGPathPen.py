from typing import Callable
from fontTools.pens.basePen import BasePen
class SVGPathPen(BasePen):
    """Pen to draw SVG path d commands.

    Example::
        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((0, 0))
        >>> pen.lineTo((1, 1))
        >>> pen.curveTo((2, 2), (3, 3), (4, 4))
        >>> pen.closePath()
        >>> pen.getCommands()
        'M0 0 1 1C2 2 3 3 4 4Z'

    Args:
        glyphSet: a dictionary of drawable glyph objects keyed by name
            used to resolve component references in composite glyphs.
        ntos: a callable that takes a number and returns a string, to
            customize how numbers are formatted (default: str).

    Note:
        Fonts have a coordinate system where Y grows up, whereas in SVG,
        Y grows down.  As such, rendering path data from this pen in
        SVG typically results in upside-down glyphs.  You can fix this
        by wrapping the data from this pen in an SVG group element with
        transform, or wrap this pen in a transform pen.  For example:

            spen = svgPathPen.SVGPathPen(glyphset)
            pen= TransformPen(spen , (1, 0, 0, -1, 0, 0))
            glyphset[glyphname].draw(pen)
            print(tpen.getCommands())
    """

    def __init__(self, glyphSet, ntos: Callable[[float], str]=str):
        BasePen.__init__(self, glyphSet)
        self._commands = []
        self._lastCommand = None
        self._lastX = None
        self._lastY = None
        self._ntos = ntos

    def _handleAnchor(self):
        """
        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((0, 0))
        >>> pen.moveTo((10, 10))
        >>> pen._commands
        ['M10 10']
        """
        if self._lastCommand == 'M':
            self._commands.pop(-1)

    def _moveTo(self, pt):
        """
        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((0, 0))
        >>> pen._commands
        ['M0 0']

        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((10, 0))
        >>> pen._commands
        ['M10 0']

        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((0, 10))
        >>> pen._commands
        ['M0 10']
        """
        self._handleAnchor()
        t = 'M%s' % pointToString(pt, self._ntos)
        self._commands.append(t)
        self._lastCommand = 'M'
        self._lastX, self._lastY = pt

    def _lineTo(self, pt):
        """
        # duplicate point
        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((10, 10))
        >>> pen.lineTo((10, 10))
        >>> pen._commands
        ['M10 10']

        # vertical line
        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((10, 10))
        >>> pen.lineTo((10, 0))
        >>> pen._commands
        ['M10 10', 'V0']

        # horizontal line
        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((10, 10))
        >>> pen.lineTo((0, 10))
        >>> pen._commands
        ['M10 10', 'H0']

        # basic
        >>> pen = SVGPathPen(None)
        >>> pen.lineTo((70, 80))
        >>> pen._commands
        ['L70 80']

        # basic following a moveto
        >>> pen = SVGPathPen(None)
        >>> pen.moveTo((0, 0))
        >>> pen.lineTo((10, 10))
        >>> pen._commands
        ['M0 0', ' 10 10']
        """
        x, y = pt
        if x == self._lastX and y == self._lastY:
            return
        elif x == self._lastX:
            cmd = 'V'
            pts = self._ntos(y)
        elif y == self._lastY:
            cmd = 'H'
            pts = self._ntos(x)
        elif self._lastCommand == 'M':
            cmd = None
            pts = ' ' + pointToString(pt, self._ntos)
        else:
            cmd = 'L'
            pts = pointToString(pt, self._ntos)
        t = ''
        if cmd:
            t += cmd
            self._lastCommand = cmd
        t += pts
        self._commands.append(t)
        self._lastX, self._lastY = pt

    def _curveToOne(self, pt1, pt2, pt3):
        """
        >>> pen = SVGPathPen(None)
        >>> pen.curveTo((10, 20), (30, 40), (50, 60))
        >>> pen._commands
        ['C10 20 30 40 50 60']
        """
        t = 'C'
        t += pointToString(pt1, self._ntos) + ' '
        t += pointToString(pt2, self._ntos) + ' '
        t += pointToString(pt3, self._ntos)
        self._commands.append(t)
        self._lastCommand = 'C'
        self._lastX, self._lastY = pt3

    def _qCurveToOne(self, pt1, pt2):
        """
        >>> pen = SVGPathPen(None)
        >>> pen.qCurveTo((10, 20), (30, 40))
        >>> pen._commands
        ['Q10 20 30 40']
        >>> from fontTools.misc.roundTools import otRound
        >>> pen = SVGPathPen(None, ntos=lambda v: str(otRound(v)))
        >>> pen.qCurveTo((3, 3), (7, 5), (11, 4))
        >>> pen._commands
        ['Q3 3 5 4', 'Q7 5 11 4']
        """
        assert pt2 is not None
        t = 'Q'
        t += pointToString(pt1, self._ntos) + ' '
        t += pointToString(pt2, self._ntos)
        self._commands.append(t)
        self._lastCommand = 'Q'
        self._lastX, self._lastY = pt2

    def _closePath(self):
        """
        >>> pen = SVGPathPen(None)
        >>> pen.closePath()
        >>> pen._commands
        ['Z']
        """
        self._commands.append('Z')
        self._lastCommand = 'Z'
        self._lastX = self._lastY = None

    def _endPath(self):
        """
        >>> pen = SVGPathPen(None)
        >>> pen.endPath()
        >>> pen._commands
        []
        """
        self._lastCommand = None
        self._lastX = self._lastY = None

    def getCommands(self):
        return ''.join(self._commands)