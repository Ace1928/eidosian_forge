import numpy as np
from ... import Point, PolyLineROI
from ... import functions as pgfn
from ... import metaarray as metaarray
from . import functions
from .common import CtrlNode, PlottingCtrlNode, metaArrayWrapper
class RemoveBaseline(PlottingCtrlNode):
    """Remove an arbitrary, graphically defined baseline from the data."""
    nodeName = 'RemoveBaseline'

    def __init__(self, name):
        PlottingCtrlNode.__init__(self, name)
        self.line = PolyLineROI([[0, 0], [1, 0]])
        self.line.sigRegionChanged.connect(self.changed)

    def connectToPlot(self, node):
        """Define what happens when the node is connected to a plot"""
        if node.plot is None:
            return
        node.getPlot().addItem(self.line)

    def disconnectFromPlot(self, plot):
        """Define what happens when the node is disconnected from a plot"""
        plot.removeItem(self.line)

    def processData(self, data):
        h0 = self.line.getHandles()[0]
        h1 = self.line.getHandles()[-1]
        timeVals = data.xvals(0)
        h0.setPos(timeVals[0], h0.pos()[1])
        h1.setPos(timeVals[-1], h1.pos()[1])
        pts = self.line.listPoints()
        pts, indices = self.adjustXPositions(pts, timeVals)
        arr = np.zeros(len(data), dtype=float)
        n = 1
        arr[0] = pts[0].y()
        for i in range(len(pts) - 1):
            x1 = pts[i].x()
            x2 = pts[i + 1].x()
            y1 = pts[i].y()
            y2 = pts[i + 1].y()
            m = (y2 - y1) / (x2 - x1)
            b = y1
            times = timeVals[(timeVals > x1) * (timeVals <= x2)]
            arr[n:n + len(times)] = m * (times - times[0]) + b
            n += len(times)
        return data - arr

    def adjustXPositions(self, pts, data):
        """Return a list of Point() where the x position is set to the nearest x value in *data* for each point in *pts*."""
        points = []
        timeIndices = []
        for p in pts:
            x = np.argwhere(abs(data - p.x()) == abs(data - p.x()).min())
            points.append(Point(data[x], p.y()))
            timeIndices.append(x)
        return (points, timeIndices)