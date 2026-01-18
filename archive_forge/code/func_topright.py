import doctest
import collections
@topright.setter
def topright(self, value):
    if self._readOnly:
        raise PyRectException('Rect object is read-only')
    _checkForTwoIntOrFloatTuple(value)
    newRight, newTop = value
    if newRight != self._left + self._width or newTop != self._top:
        originalLeft = self._left
        originalTop = self._top
        if self._enableFloat:
            self._left = newRight - self._width
            self._top = newTop
        else:
            self._left = int(newRight) - self._width
            self._top = int(newTop)
        self.callOnChange(originalLeft, originalTop, self._width, self._height)