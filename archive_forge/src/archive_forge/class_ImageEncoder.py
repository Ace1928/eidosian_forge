import os.path
from pyglet.util import CodecRegistry, Decoder, Encoder, DecodeException, EncodeException
from pyglet import compat_platform
class ImageEncoder(Encoder):

    def encode(self, image, filename, file):
        """Encode the given image to the given file.  filename
        provides a hint to the file format desired.
        """
        raise NotImplementedError()

    def __repr__(self):
        return '{0}{1}'.format(self.__class__.__name__, self.get_file_extensions())