import io
import zlib
from joblib.backports import LooseVersion
class LZ4CompressorWrapper(CompressorWrapper):
    prefix = _LZ4_PREFIX
    extension = '.lz4'

    def __init__(self):
        if lz4 is not None:
            self.fileobj_factory = LZ4FrameFile
        else:
            self.fileobj_factory = None

    def _check_versions(self):
        if lz4 is None:
            raise ValueError(LZ4_NOT_INSTALLED_ERROR)
        lz4_version = lz4.__version__
        if lz4_version.startswith('v'):
            lz4_version = lz4_version[1:]
        if LooseVersion(lz4_version) < LooseVersion('0.19'):
            raise ValueError(LZ4_NOT_INSTALLED_ERROR)

    def compressor_file(self, fileobj, compresslevel=None):
        """Returns an instance of a compressor file object."""
        self._check_versions()
        if compresslevel is None:
            return self.fileobj_factory(fileobj, 'wb')
        else:
            return self.fileobj_factory(fileobj, 'wb', compression_level=compresslevel)

    def decompressor_file(self, fileobj):
        """Returns an instance of a decompressor file object."""
        self._check_versions()
        return self.fileobj_factory(fileobj, 'rb')