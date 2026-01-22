import numpy
from . import ClusterUtils
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