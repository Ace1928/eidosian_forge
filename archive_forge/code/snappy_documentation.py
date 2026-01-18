from __future__ import absolute_import
import struct
import cramjam
Decompress 'data', returning a string containing the uncompressed
        data corresponding to at least part of the data in string. This data
        should be concatenated to the output produced by any preceding calls to
        the decompress() method. Some of the input data may be preserved in
        internal buffers for later processing.
        