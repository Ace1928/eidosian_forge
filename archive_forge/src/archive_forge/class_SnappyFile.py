from zipfile import ZipFile
import fsspec.utils
from fsspec.spec import AbstractBufferedFile
class SnappyFile(AbstractBufferedFile):

    def __init__(self, infile, mode, **kwargs):
        import snappy
        super().__init__(fs=None, path='snappy', mode=mode.strip('b') + 'b', size=999999999, **kwargs)
        self.infile = infile
        if 'r' in mode:
            self.codec = snappy.StreamDecompressor()
        else:
            self.codec = snappy.StreamCompressor()

    def _upload_chunk(self, final=False):
        self.buffer.seek(0)
        out = self.codec.add_chunk(self.buffer.read())
        self.infile.write(out)
        return True

    def seek(self, loc, whence=0):
        raise NotImplementedError('SnappyFile is not seekable')

    def seekable(self):
        return False

    def _fetch_range(self, start, end):
        """Get the specified set of bytes from remote"""
        data = self.infile.read(end - start)
        return self.codec.decompress(data)