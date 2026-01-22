from array import array
from bisect import bisect_right
from os.path import isfile
from struct import Struct
from warnings import warn
from pyzstd.zstdfile import ZstdDecompressReader, ZstdFile, \
class SeekableFormatError(Exception):
    """An error related to Zstandard Seekable Format."""

    def __init__(self, msg):
        super().__init__('Zstandard Seekable Format error: ' + msg)