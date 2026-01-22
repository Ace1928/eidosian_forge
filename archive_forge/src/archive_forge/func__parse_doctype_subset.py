import re
def _parse_doctype_subset(self, i, declstartpos):
    rawdata = self.rawdata
    n = len(rawdata)
    j = i
    while j < n:
        c = rawdata[j]
        if c == '<':
            s = rawdata[j:j + 2]
            if s == '<':
                return -1
            if s != '<!':
                self.updatepos(declstartpos, j + 1)
                raise AssertionError('unexpected char in internal subset (in %r)' % s)
            if j + 2 == n:
                return -1
            if j + 4 > n:
                return -1
            if rawdata[j:j + 4] == '<!--':
                j = self.parse_comment(j, report=0)
                if j < 0:
                    return j
                continue
            name, j = self._scan_name(j + 2, declstartpos)
            if j == -1:
                return -1
            if name not in {'attlist', 'element', 'entity', 'notation'}:
                self.updatepos(declstartpos, j + 2)
                raise AssertionError('unknown declaration %r in internal subset' % name)
            meth = getattr(self, '_parse_doctype_' + name)
            j = meth(j, declstartpos)
            if j < 0:
                return j
        elif c == '%':
            if j + 1 == n:
                return -1
            s, j = self._scan_name(j + 1, declstartpos)
            if j < 0:
                return j
            if rawdata[j] == ';':
                j = j + 1
        elif c == ']':
            j = j + 1
            while j < n and rawdata[j].isspace():
                j = j + 1
            if j < n:
                if rawdata[j] == '>':
                    return j
                self.updatepos(declstartpos, j)
                raise AssertionError('unexpected char after internal subset')
            else:
                return -1
        elif c.isspace():
            j = j + 1
        else:
            self.updatepos(declstartpos, j)
            raise AssertionError('unexpected char %r in internal subset' % c)
    return -1