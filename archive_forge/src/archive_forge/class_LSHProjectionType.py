import flatbuffers
from flatbuffers.compat import import_numpy
class LSHProjectionType(object):
    UNKNOWN = 0
    SPARSE = 1
    DENSE = 2