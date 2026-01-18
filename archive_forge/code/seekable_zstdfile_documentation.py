from array import array
from bisect import bisect_right
from os.path import isfile
from struct import Struct
from warnings import warn
from pyzstd.zstdfile import ZstdDecompressReader, ZstdFile, \
Check if a file is Zstandard Seekable Format file or 0-size file.

        It parses the seek table at the end of the file, returns True if no
        format error.

        filename can be either a file path (str/bytes/PathLike), or can be an
        existing file object in reading mode.
        