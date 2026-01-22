import socket
import struct
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, List, Optional
from wandb.proto import wandb_server_pb2 as spb
from . import tracelog
class SockBuffer:
    _buf_list: List[bytes]
    _buf_lengths: List[int]
    _buf_total: int

    def __init__(self) -> None:
        self._buf_list = []
        self._buf_lengths = []
        self._buf_total = 0

    @property
    def length(self) -> int:
        return self._buf_total

    def _get(self, start: int, end: int, peek: bool=False) -> bytes:
        index: Optional[int] = None
        buffers = []
        need = end
        for i, (buf_len, buf_data) in enumerate(zip(self._buf_lengths, self._buf_list)):
            buffers.append(buf_data[:need] if need < buf_len else buf_data)
            if need <= buf_len:
                index = i
                break
            need -= buf_len
        if index is None:
            raise IndexError('SockBuffer index out of range')
        if not peek:
            self._buf_total -= end
            if need < buf_len:
                self._buf_list = self._buf_list[index:]
                self._buf_lengths = self._buf_lengths[index:]
                self._buf_list[0] = self._buf_list[0][need:]
                self._buf_lengths[0] -= need
            else:
                self._buf_list = self._buf_list[index + 1:]
                self._buf_lengths = self._buf_lengths[index + 1:]
        return b''.join(buffers)[start:end]

    def get(self, start: int, end: int) -> bytes:
        return self._get(start, end)

    def peek(self, start: int, end: int) -> bytes:
        return self._get(start, end, peek=True)

    def put(self, data: bytes, data_len: int) -> None:
        self._buf_list.append(data)
        self._buf_lengths.append(data_len)
        self._buf_total += data_len