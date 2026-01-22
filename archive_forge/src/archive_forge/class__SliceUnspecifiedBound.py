import dns.exception
from ._compat import binary_type, string_types, PY2
class _SliceUnspecifiedBound(binary_type):

    def __getitem__(self, key):
        return key.stop
    if PY2:

        def __getslice__(self, i, j):
            return self.__getitem__(slice(i, j))