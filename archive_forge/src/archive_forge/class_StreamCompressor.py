from __future__ import absolute_import
import struct
import cramjam
class StreamCompressor:
    """This class implements the compressor-side of the proposed Snappy framing
    format, found at

        http://code.google.com/p/snappy/source/browse/trunk/framing_format.txt
            ?spec=svn68&r=71

    This class matches the interface found for the zlib module's compression
    objects (see zlib.compressobj), but also provides some additions, such as
    the snappy framing format's ability to intersperse uncompressed data.

    Keep in mind that this compressor object does no buffering for you to
    appropriately size chunks. Every call to StreamCompressor.compress results
    in a unique call to the underlying snappy compression method.
    """

    def __init__(self):
        self.c = cramjam.snappy.Compressor()

    def add_chunk(self, data: bytes, compress=None):
        """Add a chunk, returning a string that is framed and compressed. 
        
        Outputs a single snappy chunk; if it is the very start of the stream,
        will also contain the stream header chunk.
        """
        self.c.compress(data)
        return self.flush()
    compress = add_chunk

    def flush(self):
        return bytes(self.c.flush())

    def copy(self):
        """This method exists for compatibility with the zlib compressobj.
        """
        return self