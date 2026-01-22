from rdkit.sping.colors import *
def arcPoints(self, x1, y1, x2, y2, startAng=0, extent=360):
    """Return a list of points approximating the given arc."""
    xScale = abs((x2 - x1) / 2.0)
    yScale = abs((y2 - y1) / 2.0)
    x = min(x1, x2) + xScale
    y = min(y1, y2) + yScale
    steps = min(max(xScale, yScale) * (extent / 10.0) / 10, 200)
    if steps < 5:
        steps = 5
    from math import cos, pi, sin
    pointlist = []
    step = float(extent) / steps
    angle = startAng
    for i in range(int(steps + 1)):
        point = (x + xScale * cos(angle / 180.0 * pi), y - yScale * sin(angle / 180.0 * pi))
        pointlist.append(point)
        angle = angle + step
    return pointlist