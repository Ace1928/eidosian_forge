import logging
import struct
import sys
from io import BytesIO
from typing import (
from . import settings
from .cmapdb import CMap
from .cmapdb import CMapBase
from .cmapdb import CMapDB
from .cmapdb import CMapParser
from .cmapdb import FileUnicodeMap
from .cmapdb import IdentityUnicodeMap
from .cmapdb import UnicodeMap
from .encodingdb import EncodingDB
from .encodingdb import name2unicode
from .fontmetrics import FONT_METRICS
from .pdftypes import PDFException
from .pdftypes import PDFStream
from .pdftypes import dict_value
from .pdftypes import int_value
from .pdftypes import list_value
from .pdftypes import num_value
from .pdftypes import resolve1, resolve_all
from .pdftypes import stream_value
from .psparser import KWD
from .psparser import LIT
from .psparser import PSEOF
from .psparser import PSKeyword
from .psparser import PSLiteral
from .psparser import PSStackParser
from .psparser import literal_name
from .utils import Matrix, Point
from .utils import Rect
from .utils import apply_matrix_norm
from .utils import choplist
from .utils import nunpack
class TrueTypeFont:

    class CMapNotFound(Exception):
        pass

    def __init__(self, name: str, fp: BinaryIO) -> None:
        self.name = name
        self.fp = fp
        self.tables: Dict[bytes, Tuple[int, int]] = {}
        self.fonttype = fp.read(4)
        try:
            ntables, _1, _2, _3 = cast(Tuple[int, int, int, int], struct.unpack('>HHHH', fp.read(8)))
            for _ in range(ntables):
                name_bytes, tsum, offset, length = cast(Tuple[bytes, int, int, int], struct.unpack('>4sLLL', fp.read(16)))
                self.tables[name_bytes] = (offset, length)
        except struct.error:
            pass
        return

    def create_unicode_map(self) -> FileUnicodeMap:
        if b'cmap' not in self.tables:
            raise TrueTypeFont.CMapNotFound
        base_offset, length = self.tables[b'cmap']
        fp = self.fp
        fp.seek(base_offset)
        version, nsubtables = cast(Tuple[int, int], struct.unpack('>HH', fp.read(4)))
        subtables: List[Tuple[int, int, int]] = []
        for i in range(nsubtables):
            subtables.append(cast(Tuple[int, int, int], struct.unpack('>HHL', fp.read(8))))
        char2gid: Dict[int, int] = {}
        for platform_id, encoding_id, st_offset in subtables:
            if not (platform_id == 0 or (platform_id == 3 and encoding_id in [1, 10])):
                continue
            fp.seek(base_offset + st_offset)
            fmttype, fmtlen, fmtlang = cast(Tuple[int, int, int], struct.unpack('>HHH', fp.read(6)))
            if fmttype == 0:
                char2gid.update(enumerate(cast(Tuple[int, ...], struct.unpack('>256B', fp.read(256)))))
            elif fmttype == 2:
                subheaderkeys = cast(Tuple[int, ...], struct.unpack('>256H', fp.read(512)))
                firstbytes = [0] * 8192
                for i, k in enumerate(subheaderkeys):
                    firstbytes[k // 8] = i
                nhdrs = max(subheaderkeys) // 8 + 1
                hdrs: List[Tuple[int, int, int, int, int]] = []
                for i in range(nhdrs):
                    firstcode, entcount, delta, offset = cast(Tuple[int, int, int, int], struct.unpack('>HHhH', fp.read(8)))
                    hdrs.append((i, firstcode, entcount, delta, fp.tell() - 2 + offset))
                for i, firstcode, entcount, delta, pos in hdrs:
                    if not entcount:
                        continue
                    first = firstcode + (firstbytes[i] << 8)
                    fp.seek(pos)
                    for c in range(entcount):
                        gid = cast(Tuple[int], struct.unpack('>H', fp.read(2)))[0]
                        if gid:
                            gid += delta
                        char2gid[first + c] = gid
            elif fmttype == 4:
                segcount, _1, _2, _3 = cast(Tuple[int, int, int, int], struct.unpack('>HHHH', fp.read(8)))
                segcount //= 2
                ecs = cast(Tuple[int, ...], struct.unpack('>%dH' % segcount, fp.read(2 * segcount)))
                fp.read(2)
                scs = cast(Tuple[int, ...], struct.unpack('>%dH' % segcount, fp.read(2 * segcount)))
                idds = cast(Tuple[int, ...], struct.unpack('>%dh' % segcount, fp.read(2 * segcount)))
                pos = fp.tell()
                idrs = cast(Tuple[int, ...], struct.unpack('>%dH' % segcount, fp.read(2 * segcount)))
                for ec, sc, idd, idr in zip(ecs, scs, idds, idrs):
                    if idr:
                        fp.seek(pos + idr)
                        for c in range(sc, ec + 1):
                            b = cast(Tuple[int], struct.unpack('>H', fp.read(2)))[0]
                            char2gid[c] = b + idd & 65535
                    else:
                        for c in range(sc, ec + 1):
                            char2gid[c] = c + idd & 65535
            else:
                assert False, str(('Unhandled', fmttype))
        if not char2gid:
            raise TrueTypeFont.CMapNotFound
        unicode_map = FileUnicodeMap()
        for char, gid in char2gid.items():
            unicode_map.add_cid2unichr(gid, char)
        return unicode_map