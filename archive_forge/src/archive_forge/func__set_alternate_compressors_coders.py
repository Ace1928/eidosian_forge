import bz2
import lzma
import struct
import sys
import zlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import bcj
import inflate64
import pyppmd
import pyzstd
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from py7zr.exceptions import PasswordRequired, UnsupportedCompressionMethodError
from py7zr.helpers import Buffer, calculate_crc32, calculate_key
from py7zr.properties import (
def _set_alternate_compressors_coders(self, alt_filter, password=None):
    filter_id = alt_filter['id']
    properties = None
    if filter_id not in algorithm_class_map:
        raise UnsupportedCompressionMethodError(filter_id, 'Unknown filter_id is given.')
    elif SupportedMethods.is_crypto_id(filter_id):
        compressor = algorithm_class_map[filter_id][0](password)
    elif SupportedMethods.need_property(filter_id):
        if filter_id == FILTER_ZSTD:
            level = alt_filter.get('level', 3)
            properties = struct.pack('BBBBB', pyzstd.zstd_version_info[0], pyzstd.zstd_version_info[1], level, 0, 0)
            compressor = algorithm_class_map[filter_id][0](level=level)
        elif filter_id == FILTER_PPMD:
            properties = PpmdCompressor.encode_filter_properties(alt_filter)
            compressor = algorithm_class_map[filter_id][0](properties)
        elif filter_id == FILTER_BROTLI:
            level = alt_filter.get('level', 11)
            properties = struct.pack('BBB', brotli_major, brotli_minor, level)
            compressor = algorithm_class_map[filter_id][0](level)
    else:
        compressor = algorithm_class_map[filter_id][0]()
    if SupportedMethods.is_crypto_id(filter_id):
        properties = compressor.encode_filter_properties()
    self.chain.append(compressor)
    self._unpacksizes.append(0)
    self.coders.insert(0, {'method': SupportedMethods.get_method_id(filter_id), 'properties': properties, 'numinstreams': 1, 'numoutstreams': 1})