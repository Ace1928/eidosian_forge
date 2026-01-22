import re
def _parse_rect(self, rect):
    x = float(rect.attrib.get('x', 0))
    y = float(rect.attrib.get('y', 0))
    w = float(rect.attrib.get('width'))
    h = float(rect.attrib.get('height'))
    rx = float(rect.attrib.get('rx', 0))
    ry = float(rect.attrib.get('ry', 0))
    rx = _prefer_non_zero(rx, ry)
    ry = _prefer_non_zero(ry, rx)
    self._start_path()
    self.M(x + rx, y)
    self.H(x + w - rx)
    if rx > 0:
        self.A(rx, ry, x + w, y + ry)
    self.V(y + h - ry)
    if rx > 0:
        self.A(rx, ry, x + w - rx, y + h)
    self.H(x + rx)
    if rx > 0:
        self.A(rx, ry, x, y + h - ry)
    self.V(y + ry)
    if rx > 0:
        self.A(rx, ry, x + rx, y)
    self._end_path()