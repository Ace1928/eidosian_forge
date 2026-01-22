import ipaddress
import struct
from ipaddress import IPv4Address, IPv6Address
from os import PathLike
from typing import Any, AnyStr, Dict, IO, List, Optional, Tuple, Union
from maxminddb.const import MODE_AUTO, MODE_MMAP, MODE_FILE, MODE_MEMORY, MODE_FD
from maxminddb.decoder import Decoder
from maxminddb.errors import InvalidDatabaseError
from maxminddb.file import FileBuffer
from maxminddb.types import Record
def _resolve_data_pointer(self, pointer: int) -> Record:
    resolved = pointer - self._metadata.node_count + self._metadata.search_tree_size
    if resolved >= self._buffer_size:
        raise InvalidDatabaseError("The MaxMind DB file's search tree is corrupt")
    data, _ = self._decoder.decode(resolved)
    return data