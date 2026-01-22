import flatbuffers
from flatbuffers.compat import import_numpy
@classmethod
def Int32VectorBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
    return flatbuffers.util.BufferHasIdentifier(buf, offset, b'TFL3', size_prefixed=size_prefixed)