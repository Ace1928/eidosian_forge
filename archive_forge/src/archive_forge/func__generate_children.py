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
def _generate_children(self, node, depth, ip_acc):
    if ip_acc != 0 and node == self._ipv4_start:
        return
    node_count = self._metadata.node_count
    if node > node_count:
        bits = 128 if self._metadata.ip_version == 6 else 32
        ip_acc <<= bits - depth
        if ip_acc <= _IPV4_MAX_NUM and bits == 128:
            depth -= 96
        yield (ipaddress.ip_network((ip_acc, depth)), self._resolve_data_pointer(node))
    elif node < node_count:
        left = self._read_node(node, 0)
        ip_acc <<= 1
        depth += 1
        yield from self._generate_children(left, depth, ip_acc)
        right = self._read_node(node, 1)
        yield from self._generate_children(right, depth, ip_acc | 1)