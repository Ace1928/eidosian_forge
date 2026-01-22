from pyglet.util import CodecRegistry, Decoder, Encoder, DecodeException, EncodeException
class ModelDecoder(Decoder):

    def decode(self, filename, file, batch, group):
        """Decode the given file object and return an instance of `Model`.
        Throws ModelDecodeException if there is an error.  filename
        can be a file type hint.
        """
        raise NotImplementedError()