from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
def _write_stream(stream, length, data):
    if length < 0:
        raise ValueError('length must be >= 0', length)
    if len(data) != length:
        raise FieldError('expected %d, found %d' % (length, len(data)))
    stream.write(data)