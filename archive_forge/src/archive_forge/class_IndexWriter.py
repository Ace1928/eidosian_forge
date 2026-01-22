import zlib
from gitdb.exc import (
from gitdb.util import (
from gitdb.fun import (
from gitdb.base import (      # Amazing !
from gitdb.stream import (
from struct import pack
from binascii import crc32
from gitdb.const import NULL_BYTE
import tempfile
import array
import os
import sys
class IndexWriter:
    """Utility to cache index information, allowing to write all information later
    in one go to the given stream
    **Note:** currently only writes v2 indices"""
    __slots__ = '_objs'

    def __init__(self):
        self._objs = list()

    def append(self, binsha, crc, offset):
        """Append one piece of object information"""
        self._objs.append((binsha, crc, offset))

    def write(self, pack_sha, write):
        """Write the index file using the given write method
        :param pack_sha: binary sha over the whole pack that we index
        :return: sha1 binary sha over all index file contents"""
        self._objs.sort(key=lambda o: o[0])
        sha_writer = FlexibleSha1Writer(write)
        sha_write = sha_writer.write
        sha_write(PackIndexFile.index_v2_signature)
        sha_write(pack('>L', PackIndexFile.index_version_default))
        tmplist = list((0,) * 256)
        for t in self._objs:
            tmplist[byte_ord(t[0][0])] += 1
        for i in range(255):
            v = tmplist[i]
            sha_write(pack('>L', v))
            tmplist[i + 1] += v
        sha_write(pack('>L', tmplist[255]))
        sha_write(b''.join((t[0] for t in self._objs)))
        for t in self._objs:
            sha_write(pack('>L', t[1] & 4294967295))
        tmplist = list()
        for t in self._objs:
            ofs = t[2]
            if ofs > 2147483647:
                tmplist.append(ofs)
                ofs = 2147483648 + len(tmplist) - 1
            sha_write(pack('>L', ofs & 4294967295))
        for ofs in tmplist:
            sha_write(pack('>Q', ofs))
        assert len(pack_sha) == 20
        sha_write(pack_sha)
        sha = sha_writer.sha(as_hex=False)
        write(sha)
        return sha