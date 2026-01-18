import doctest
import collections
def unionAll(self, otherRects):
    """Adjusts the width and height to also cover all the `Rect` objects in
        the `otherRects` sequence.

        >>> r = Rect(0, 0, 100, 100)
        >>> r1 = Rect(0, 0, 150, 100)
        >>> r2 = Rect(-10, -10, 100, 100)
        >>> r.unionAll([r1, r2])
        >>> r
        Rect(left=-10, top=-10, width=160, height=110)
        """
    otherRects = list(otherRects)
    otherRects.append(self)
    unionLeft = min([r._left for r in otherRects])
    unionTop = min([r._top for r in otherRects])
    unionRight = max([r.right for r in otherRects])
    unionBottom = max([r.bottom for r in otherRects])
    self._left = unionLeft
    self._top = unionTop
    self._width = unionRight - unionLeft
    self._height = unionBottom - unionTop