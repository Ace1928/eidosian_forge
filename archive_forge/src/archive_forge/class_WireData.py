import dns.exception
from ._compat import binary_type, string_types, PY2
class WireData(binary_type):

    def __getitem__(self, key):
        try:
            if isinstance(key, slice):
                start = key.start
                stop = key.stop
                if PY2:
                    if stop == _unspecified_bound:
                        stop = len(self)
                    if start < 0 or stop < 0:
                        raise dns.exception.FormError
                    if start != stop:
                        super(WireData, self).__getitem__(start)
                        super(WireData, self).__getitem__(stop - 1)
                else:
                    for index in (start, stop):
                        if index is None:
                            continue
                        elif abs(index) > len(self):
                            raise dns.exception.FormError
                return WireData(super(WireData, self).__getitem__(slice(start, stop)))
            return bytearray(self.unwrap())[key]
        except IndexError:
            raise dns.exception.FormError
    if PY2:

        def __getslice__(self, i, j):
            return self.__getitem__(slice(i, j))

    def __iter__(self):
        i = 0
        while 1:
            try:
                yield self[i]
                i += 1
            except dns.exception.FormError:
                raise StopIteration

    def unwrap(self):
        return binary_type(self)