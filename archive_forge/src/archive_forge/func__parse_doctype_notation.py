import re
def _parse_doctype_notation(self, i, declstartpos):
    name, j = self._scan_name(i, declstartpos)
    if j < 0:
        return j
    rawdata = self.rawdata
    while 1:
        c = rawdata[j:j + 1]
        if not c:
            return -1
        if c == '>':
            return j + 1
        if c in '\'"':
            m = _declstringlit_match(rawdata, j)
            if not m:
                return -1
            j = m.end()
        else:
            name, j = self._scan_name(j, declstartpos)
            if j < 0:
                return j