import doctest
import collections
def _getRectsAndPoints(rectsOrPoints):
    points = []
    rects = []
    for rectOrPoint in rectsOrPoints:
        try:
            _checkForTwoIntOrFloatTuple(rectOrPoint)
            points.append(rectOrPoint)
        except PyRectException:
            try:
                _checkForFourIntOrFloatTuple(rectOrPoint)
            except:
                raise PyRectException('argument is not a point or a rect tuple')
            rects.append(rectOrPoint)
    return (rects, points)