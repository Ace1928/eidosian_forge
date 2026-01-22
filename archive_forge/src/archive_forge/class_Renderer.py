from reportlab.graphics.shapes import *
from reportlab.lib.validators import DerivedValue
from reportlab import rl_config
from . transform import mmult, inverse
class Renderer:
    """Virtual superclass for graphics renderers."""

    def undefined(self, operation):
        raise ValueError('%s operation not defined at superclass class=%s' % (operation, self.__class__))

    def draw(self, drawing, canvas, x=0, y=0, showBoundary=rl_config._unset_):
        """This is the top level function, which draws the drawing at the given
        location. The recursive part is handled by drawNode."""
        self._tracker = StateTracker(defaultObj=drawing)
        if showBoundary is rl_config._unset_:
            showBoundary = rl_config.showBoundary
        self._canvas = canvas
        canvas.__dict__['_drawing'] = self._drawing = drawing
        drawing._parent = None
        try:
            if showBoundary:
                if hasattr(canvas, 'drawBoundary'):
                    canvas.drawBoundary(showBoundary, x, y, drawing.width, drawing.height)
                else:
                    canvas.rect(x, y, drawing.width, drawing.height)
            canvas.saveState()
            self.initState(x, y)
            self.drawNode(drawing)
            self.pop()
            canvas.restoreState()
        finally:
            del self._canvas, self._drawing, canvas._drawing, drawing._parent, self._tracker

    def initState(self, x, y):
        deltas = self._tracker._combined[-1]
        deltas['transform'] = tuple(list(deltas['transform'])[:4]) + (x, y)
        self._tracker.push(deltas)
        self.applyStateChanges(deltas, {})

    def pop(self):
        self._tracker.pop()

    def drawNode(self, node):
        """This is the recursive method called for each node
        in the tree"""
        self.undefined('drawNode')

    def getStateValue(self, key):
        """Return current state parameter for given key"""
        currentState = self._tracker._combined[-1]
        return currentState[key]

    def fillDerivedValues(self, node):
        """Examine a node for any values which are Derived,
        and replace them with their calculated values.
        Generally things may look at the drawing or their
        parent.

        """
        for key, value in node.__dict__.items():
            if isinstance(value, DerivedValue):
                newValue = value.getValue(self, key)
                node.__dict__[key] = newValue

    def drawNodeDispatcher(self, node):
        """dispatch on the node's (super) class: shared code"""
        canvas = getattr(self, '_canvas', None)
        try:
            node = _expandUserNode(node, canvas)
            if not node:
                return
            if hasattr(node, '_canvas'):
                ocanvas = 1
            else:
                node._canvas = canvas
                ocanvas = None
            self.fillDerivedValues(node)
            dtcb = getattr(node, '_drawTimeCallback', None)
            if dtcb:
                dtcb(node, canvas=canvas, renderer=self)
            if isinstance(node, Line):
                self.drawLine(node)
            elif isinstance(node, Image):
                self.drawImage(node)
            elif isinstance(node, Rect):
                self.drawRect(node)
            elif isinstance(node, Circle):
                self.drawCircle(node)
            elif isinstance(node, Ellipse):
                self.drawEllipse(node)
            elif isinstance(node, PolyLine):
                self.drawPolyLine(node)
            elif isinstance(node, Polygon):
                self.drawPolygon(node)
            elif isinstance(node, Path):
                self.drawPath(node)
            elif isinstance(node, String):
                self.drawString(node)
            elif isinstance(node, Group):
                self.drawGroup(node)
            elif isinstance(node, Wedge):
                self.drawWedge(node)
            elif isinstance(node, DirectDraw):
                node.drawDirectly(self)
            else:
                print('DrawingError', 'Unexpected element %s in drawing!' % str(node))
        finally:
            if not ocanvas:
                del node._canvas
    _restores = {'stroke': '_stroke', 'stroke_width': '_lineWidth', 'stroke_linecap': '_lineCap', 'stroke_linejoin': '_lineJoin', 'fill': '_fill', 'font_family': '_font', 'font_size': '_fontSize'}

    def drawGroup(self, group):
        canvas = getattr(self, '_canvas', None)
        for node in group.getContents():
            node = _expandUserNode(node, canvas)
            if not node:
                continue
            self.fillDerivedValues(node)
            try:
                if hasattr(node, '_canvas'):
                    ocanvas = 1
                else:
                    node._canvas = canvas
                    ocanvas = None
                node._parent = group
                self.drawNode(node)
            finally:
                del node._parent
                if not ocanvas:
                    del node._canvas

    def drawWedge(self, wedge):
        P = wedge.asPolygon()
        if isinstance(P, Path):
            self.drawPath(P)
        else:
            self.drawPolygon(P)

    def drawPath(self, path):
        polygons = path.asPolygons()
        for polygon in polygons:
            self.drawPolygon(polygon)

    def drawRect(self, rect):
        self.undefined('drawRect')

    def drawLine(self, line):
        self.undefined('drawLine')

    def drawCircle(self, circle):
        self.undefined('drawCircle')

    def drawPolyLine(self, p):
        self.undefined('drawPolyLine')

    def drawEllipse(self, ellipse):
        self.undefined('drawEllipse')

    def drawPolygon(self, p):
        self.undefined('drawPolygon')

    def drawString(self, stringObj):
        self.undefined('drawString')

    def applyStateChanges(self, delta, newState):
        """This takes a set of states, and outputs the operators
        needed to set those properties"""
        self.undefined('applyStateChanges')

    def drawImage(self, *args, **kwds):
        raise NotImplementedError('drawImage')