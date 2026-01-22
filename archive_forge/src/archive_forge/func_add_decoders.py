import os.path
from pyglet.util import CodecRegistry, Decoder, Encoder, DecodeException, EncodeException
from pyglet import compat_platform
def add_decoders(self, module):
    """Override the default method to also add animation decoders.
        """
    super().add_decoders(module)
    for decoder in module.get_decoders():
        for extension in decoder.get_animation_file_extensions():
            if extension not in self._decoder_animation_extensions:
                self._decoder_animation_extensions[extension] = []
            self._decoder_animation_extensions[extension].append(decoder)