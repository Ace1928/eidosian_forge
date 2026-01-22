import os.path
from pyglet.util import CodecRegistry, Decoder, Encoder, DecodeException, EncodeException
from pyglet import compat_platform
class _ImageCodecRegistry(CodecRegistry):
    """Subclass of CodecRegistry that adds support for animation methods."""

    def __init__(self):
        self._decoder_animation_extensions = {}
        super().__init__()

    def add_decoders(self, module):
        """Override the default method to also add animation decoders.
        """
        super().add_decoders(module)
        for decoder in module.get_decoders():
            for extension in decoder.get_animation_file_extensions():
                if extension not in self._decoder_animation_extensions:
                    self._decoder_animation_extensions[extension] = []
                self._decoder_animation_extensions[extension].append(decoder)

    def get_animation_decoders(self, filename=None):
        """Get a list of animation decoders. If a `filename` is provided, only
           decoders supporting that extension will be returned. An empty list
           will be return if no encoders for that extension are available.
        """
        if filename:
            extension = os.path.splitext(filename)[1].lower()
            return self._decoder_animation_extensions.get(extension, [])
        return self._decoders

    def decode_animation(self, filename, file, **kwargs):
        first_exception = None
        for decoder in self.get_animation_decoders(filename):
            try:
                return decoder.decode_animation(filename, file, **kwargs)
            except DecodeException as e:
                if not first_exception:
                    first_exception = e
                if file:
                    file.seek(0)
        for decoder in self.get_animation_decoders():
            try:
                return decoder.decode_animation(filename, file, **kwargs)
            except DecodeException:
                if file:
                    file.seek(0)
        if not first_exception:
            raise DecodeException(f'No decoders available for this file type: {filename}')
        raise first_exception