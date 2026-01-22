from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
class Buffered(Subconstruct):
    """
    Creates an in-memory buffered stream, which can undergo encoding and
    decoding prior to being passed on to the subconstruct.
    See also Bitwise.

    Note:
    * Do not use pointers inside Buffered

    Parameters:
    * subcon - the subcon which will operate on the buffer
    * encoder - a function that takes a string and returns an encoded
      string (used after building)
    * decoder - a function that takes a string and returns a decoded
      string (used before parsing)
    * resizer - a function that takes the size of the subcon and "adjusts"
      or "resizes" it according to the encoding/decoding process.

    Example:
    Buffered(BitField("foo", 16),
        encoder = decode_bin,
        decoder = encode_bin,
        resizer = lambda size: size / 8,
    )
    """
    __slots__ = ['encoder', 'decoder', 'resizer']

    def __init__(self, subcon, decoder, encoder, resizer):
        Subconstruct.__init__(self, subcon)
        self.encoder = encoder
        self.decoder = decoder
        self.resizer = resizer

    def _parse(self, stream, context):
        data = _read_stream(stream, self._sizeof(context))
        stream2 = BytesIO(self.decoder(data))
        return self.subcon._parse(stream2, context)

    def _build(self, obj, stream, context):
        size = self._sizeof(context)
        stream2 = BytesIO()
        self.subcon._build(obj, stream2, context)
        data = self.encoder(stream2.getvalue())
        assert len(data) == size
        _write_stream(stream, self._sizeof(context), data)

    def _sizeof(self, context):
        return self.resizer(self.subcon._sizeof(context))