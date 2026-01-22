import warnings
from pyglet.util import CodecRegistry, Decoder, Encoder
from .base import *
import pyglet
class MediaEncoder(Encoder):

    def encode(self, source, filename, file):
        """Encode the given source to the given file.  `filename`
        provides a hint to the file format desired.  options are
        encoder-specific, and unknown options should be ignored or
        issue warnings.
        """
        raise NotImplementedError()