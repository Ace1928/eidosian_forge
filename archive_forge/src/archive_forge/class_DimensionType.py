import flatbuffers
from flatbuffers.compat import import_numpy
class DimensionType(object):
    DENSE = 0
    SPARSE_CSR = 1