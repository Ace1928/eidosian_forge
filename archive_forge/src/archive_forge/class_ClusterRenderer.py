import numpy
from . import ClusterUtils
class ClusterRenderer(object):

    def __init__(self, canvas, size, ptColors=[], lineWidth=None, showIndices=0, showNodes=1, stopAtCentroids=0, logScale=0, tooClose=-1):
        self.canvas = canvas
        self.size = size
        self.ptColors = ptColors
        self.lineWidth = lineWidth
        self.showIndices = showIndices
        self.showNodes = showNodes
        self.stopAtCentroids = stopAtCentroids
        self.logScale = logScale
        self.tooClose = tooClose

    def _AssignPointLocations(self, cluster, terminalOffset=4):
        self.pts = cluster.GetPoints()
        self.nPts = len(self.pts)
        self.xSpace = float(self.size[0] - 2 * VisOpts.xOffset) / float(self.nPts - 1)
        ySize = self.size[1]
        for i in range(self.nPts):
            pt = self.pts[i]
            if self.logScale > 0:
                v = _scaleMetric(pt.GetMetric(), self.logScale)
            else:
                v = float(pt.GetMetric())
            pt._drawPos = (VisOpts.xOffset + i * self.xSpace, ySize - (v * self.ySpace + VisOpts.yOffset) + terminalOffset)

    def _AssignClusterLocations(self, cluster):
        toDo = [cluster]
        examine = cluster.GetChildren()[:]
        while len(examine):
            node = examine.pop(0)
            children = node.GetChildren()
            if len(children):
                toDo.append(node)
                for child in children:
                    if not child.IsTerminal():
                        examine.append(child)
        toDo.reverse()
        for node in toDo:
            if self.logScale > 0:
                v = _scaleMetric(node.GetMetric(), self.logScale)
            else:
                v = float(node.GetMetric())
            childLocs = [x._drawPos[0] for x in node.GetChildren()]
            if len(childLocs):
                xp = sum(childLocs) / float(len(childLocs))
                yp = self.size[1] - (v * self.ySpace + VisOpts.yOffset)
                node._drawPos = (xp, yp)

    def _DrawToLimit(self, cluster):
        """
      we assume that _drawPos settings have been done already
    """
        if self.lineWidth is None:
            lineWidth = VisOpts.lineWidth
        else:
            lineWidth = self.lineWidth
        examine = [cluster]
        while len(examine):
            node = examine.pop(0)
            xp, yp = node._drawPos
            children = node.GetChildren()
            if abs(children[1]._drawPos[0] - children[0]._drawPos[0]) > self.tooClose:
                drawColor = VisOpts.lineColor
                self.canvas.drawLine(children[0]._drawPos[0], yp, children[-1]._drawPos[0], yp, drawColor, lineWidth)
                for child in children:
                    if self.ptColors and child.GetData() is not None:
                        drawColor = self.ptColors[child.GetData()]
                    else:
                        drawColor = VisOpts.lineColor
                    cxp, cyp = child._drawPos
                    self.canvas.drawLine(cxp, yp, cxp, cyp, drawColor, lineWidth)
                    if not child.IsTerminal():
                        examine.append(child)
                    elif self.showIndices and (not self.stopAtCentroids):
                        try:
                            txt = str(child.GetName())
                        except Exception:
                            txt = str(child.GetIndex())
                        self.canvas.drawString(txt, cxp - self.canvas.stringWidth(txt) / 2, cyp)
            else:
                self.canvas.drawLine(xp, yp, xp, self.size[1] - VisOpts.yOffset, VisOpts.hideColor, lineWidth)

    def DrawTree(self, cluster, minHeight=2.0):
        if self.logScale > 0:
            v = _scaleMetric(cluster.GetMetric(), self.logScale)
        else:
            v = float(cluster.GetMetric())
        if v <= 0:
            v = minHeight
        self.ySpace = float(self.size[1] - 2 * VisOpts.yOffset) / v
        self._AssignPointLocations(cluster)
        self._AssignClusterLocations(cluster)
        if not self.stopAtCentroids:
            self._DrawToLimit(cluster)
        else:
            raise NotImplementedError('stopAtCentroids drawing not yet implemented')