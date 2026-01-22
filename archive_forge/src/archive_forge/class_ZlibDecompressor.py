import zlib
class ZlibDecompressor:

    def __init__(self):
        self.z = zlib.decompressobj()

    def __call__(self, data):
        return self.z.decompress(data)