from paste.util import intset
import socket
def _itermasks(self, r):
    ranges = [r]
    while ranges:
        cur = ranges.pop()
        curmask = 0
        while True:
            curmasklen = 1 << 32 - curmask
            start = cur[0] + curmasklen - 1 & (1 << curmask) - 1 << 32 - curmask
            if start >= cur[0] and start + curmasklen <= cur[1]:
                break
            else:
                curmask += 1
        yield ('%s/%s' % (self._int2ip(start), curmask))
        if cur[0] < start:
            ranges.append((cur[0], start))
        if cur[1] > start + curmasklen:
            ranges.append((start + curmasklen, cur[1]))