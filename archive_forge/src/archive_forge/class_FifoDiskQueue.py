import glob
import json
import os
import sqlite3
import struct
from abc import abstractmethod
from collections import deque
from contextlib import suppress
from typing import Any, Optional
class FifoDiskQueue:
    """Persistent FIFO queue."""
    szhdr_format = '>L'
    szhdr_size = struct.calcsize(szhdr_format)

    def __init__(self, path: str, chunksize: int=100000) -> None:
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        self.info = self._loadinfo(chunksize)
        self.chunksize = self.info['chunksize']
        self.headf = self._openchunk(self.info['head'][0], 'ab+')
        self.tailf = self._openchunk(self.info['tail'][0])
        os.lseek(self.tailf.fileno(), self.info['tail'][2], os.SEEK_SET)

    def push(self, string: bytes) -> None:
        if not isinstance(string, bytes):
            raise TypeError('Unsupported type: {}'.format(type(string).__name__))
        hnum, hpos = self.info['head']
        hpos += 1
        szhdr = struct.pack(self.szhdr_format, len(string))
        os.write(self.headf.fileno(), szhdr + string)
        if hpos == self.chunksize:
            hpos = 0
            hnum += 1
            self.headf.close()
            self.headf = self._openchunk(hnum, 'ab+')
        self.info['size'] += 1
        self.info['head'] = [hnum, hpos]

    def _openchunk(self, number: int, mode: str='rb'):
        return open(os.path.join(self.path, 'q%05d' % number), mode)

    def pop(self) -> Optional[bytes]:
        tnum, tcnt, toffset = self.info['tail']
        if [tnum, tcnt] >= self.info['head']:
            return None
        tfd = self.tailf.fileno()
        szhdr = os.read(tfd, self.szhdr_size)
        if not szhdr:
            return None
        size, = struct.unpack(self.szhdr_format, szhdr)
        data = os.read(tfd, size)
        tcnt += 1
        toffset += self.szhdr_size + size
        if tcnt == self.chunksize and tnum <= self.info['head'][0]:
            tcnt = toffset = 0
            tnum += 1
            self.tailf.close()
            os.remove(self.tailf.name)
            self.tailf = self._openchunk(tnum)
        self.info['size'] -= 1
        self.info['tail'] = [tnum, tcnt, toffset]
        return data

    def peek(self) -> Optional[bytes]:
        tnum, tcnt, _ = self.info['tail']
        if [tnum, tcnt] >= self.info['head']:
            return None
        tfd = self.tailf.fileno()
        tfd_initial_pos = os.lseek(tfd, 0, os.SEEK_CUR)
        szhdr = os.read(tfd, self.szhdr_size)
        if not szhdr:
            return None
        size, = struct.unpack(self.szhdr_format, szhdr)
        data = os.read(tfd, size)
        os.lseek(tfd, tfd_initial_pos, os.SEEK_SET)
        return data

    def close(self) -> None:
        self.headf.close()
        self.tailf.close()
        self._saveinfo(self.info)
        if len(self) == 0:
            self._cleanup()

    def __len__(self) -> int:
        return self.info['size']

    def _loadinfo(self, chunksize: int) -> dict:
        infopath = self._infopath()
        if os.path.exists(infopath):
            with open(infopath) as f:
                info = json.load(f)
        else:
            info = {'chunksize': chunksize, 'size': 0, 'tail': [0, 0, 0], 'head': [0, 0]}
        return info

    def _saveinfo(self, info: dict) -> None:
        with open(self._infopath(), 'w') as f:
            json.dump(info, f)

    def _infopath(self) -> str:
        return os.path.join(self.path, 'info.json')

    def _cleanup(self) -> None:
        for x in glob.glob(os.path.join(self.path, 'q*')):
            os.remove(x)
        os.remove(os.path.join(self.path, 'info.json'))
        with suppress(OSError):
            os.rmdir(self.path)