import glob
import json
import os
import sqlite3
import struct
from abc import abstractmethod
from collections import deque
from contextlib import suppress
from typing import Any, Optional
class LifoDiskQueue:
    """Persistent LIFO queue."""
    SIZE_FORMAT = '>L'
    SIZE_SIZE = struct.calcsize(SIZE_FORMAT)

    def __init__(self, path: str) -> None:
        self.path = path
        if os.path.exists(path):
            self.f = open(path, 'rb+')
            qsize = self.f.read(self.SIZE_SIZE)
            self.size, = struct.unpack(self.SIZE_FORMAT, qsize)
            self.f.seek(0, os.SEEK_END)
        else:
            self.f = open(path, 'wb+')
            self.f.write(struct.pack(self.SIZE_FORMAT, 0))
            self.size = 0

    def push(self, string: bytes) -> None:
        if not isinstance(string, bytes):
            raise TypeError('Unsupported type: {}'.format(type(string).__name__))
        self.f.write(string)
        ssize = struct.pack(self.SIZE_FORMAT, len(string))
        self.f.write(ssize)
        self.size += 1

    def pop(self) -> Optional[bytes]:
        if not self.size:
            return None
        self.f.seek(-self.SIZE_SIZE, os.SEEK_END)
        size, = struct.unpack(self.SIZE_FORMAT, self.f.read())
        self.f.seek(-size - self.SIZE_SIZE, os.SEEK_END)
        data = self.f.read(size)
        self.f.seek(-size, os.SEEK_CUR)
        self.f.truncate()
        self.size -= 1
        return data

    def peek(self) -> Optional[bytes]:
        if not self.size:
            return None
        self.f.seek(-self.SIZE_SIZE, os.SEEK_END)
        size, = struct.unpack(self.SIZE_FORMAT, self.f.read())
        self.f.seek(-size - self.SIZE_SIZE, os.SEEK_END)
        data = self.f.read(size)
        return data

    def close(self) -> None:
        if self.size:
            self.f.seek(0)
            self.f.write(struct.pack(self.SIZE_FORMAT, self.size))
        self.f.close()
        if not self.size:
            os.remove(self.path)

    def __len__(self) -> int:
        return self.size