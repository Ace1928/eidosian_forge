import pyqtgraph as pg
def updateViews():
    global p1, p2, p3
    p2.setGeometry(p1.vb.sceneBoundingRect())
    p3.setGeometry(p1.vb.sceneBoundingRect())
    p2.linkedViewChanged(p1.vb, p2.XAxis)
    p3.linkedViewChanged(p1.vb, p3.XAxis)