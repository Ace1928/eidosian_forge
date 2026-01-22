import re
def _arc(self, c, rx, ry, x, y, large_arc):
    self._add('%s%s,%s 0 %d 1 %s,%s' % (c, _ntos(rx), _ntos(ry), large_arc, _ntos(x), _ntos(y)))