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
class SevenZipCompressor:
    """Main compressor object to configured for each 7zip folder."""
    __slots__ = ['filters', 'chain', 'compressor', 'coders', 'methods_map', 'digest', 'packsize', '_block_size', '_unpacksizes']

    def __init__(self, filters=None, password=None, blocksize: Optional[int]=None):
        self.filters: List[Dict[str, Any]] = []
        self.chain: List[ISevenZipCompressor] = []
        self.digest = 0
        self.packsize = 0
        self._unpacksizes: List[int] = []
        if blocksize:
            self._block_size = blocksize
        else:
            self._block_size = get_default_blocksize()
        if filters is None:
            self.filters = [{'id': lzma.FILTER_LZMA2, 'preset': 7 | lzma.PRESET_EXTREME}]
        else:
            self.filters = filters
        if len(self.filters) > 4:
            raise UnsupportedCompressionMethodError(filters, 'Maximum cascade of filters is 4 but got {}.'.format(len(self.filters)))
        self.methods_map = [SupportedMethods.is_native_filter(filter) for filter in self.filters]
        self.coders: List[Dict[str, Any]] = []
        if all(self.methods_map) and SupportedMethods.is_compressor(self.filters[-1]):
            self._set_native_compressors_coders(self.filters)
            return
        has_lzma2 = False
        for f in self.filters:
            if f['id'] == FILTER_LZMA2:
                has_lzma2 = True
                break
        if not has_lzma2:
            for i, f in enumerate(self.filters):
                if f['id'] == FILTER_X86 or f['id'] == FILTER_ARM or f['id'] == FILTER_ARMTHUMB or (f['id'] == FILTER_SPARC) or (f['id'] == FILTER_POWERPC):
                    self.methods_map[i] = False
        if not any(self.methods_map):
            for f in filters:
                self._set_alternate_compressors_coders(f, password)
        elif SupportedMethods.is_crypto_id(self.filters[-1]['id']) and all(self.methods_map[:-1]):
            self._set_native_compressors_coders(self.filters[:-1])
            self._set_alternate_compressors_coders(self.filters[-1], password)
        else:
            raise UnsupportedCompressionMethodError(filters, 'Unknown combination of methods.')

    def _set_native_compressors_coders(self, filters):
        self.chain.append(LZMA1Compressor(filters))
        self._unpacksizes.append(0)
        for filter in filters:
            self.coders.insert(0, SupportedMethods.get_coder(filter))

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

    def compress(self, fd, fp, crc=0):
        data = fd.read(self._block_size)
        insize = len(data)
        foutsize = 0
        while data:
            crc = calculate_crc32(data, crc)
            for i, compressor in enumerate(self.chain):
                self._unpacksizes[i] += len(data)
                data = compressor.compress(data)
            self.packsize += len(data)
            self.digest = calculate_crc32(data, self.digest)
            foutsize += len(data)
            fp.write(data)
            data = fd.read(self._block_size)
            insize += len(data)
        return (insize, foutsize, crc)

    def flush(self, fp):
        data = None
        for i, compressor in enumerate(self.chain):
            if data:
                self._unpacksizes[i] += len(data)
                data = compressor.compress(data)
                data += compressor.flush()
            else:
                data = compressor.flush()
        if data is None:
            return 0
        self.packsize += len(data)
        self.digest = calculate_crc32(data, self.digest)
        fp.write(data)
        return len(data)

    @property
    def unpacksizes(self) -> List[int]:
        result: List[int] = []
        shift = 0
        prev = False
        for i, r in enumerate(self.methods_map):
            shift += 1 if r and prev else 0
            prev = r
            result.insert(0, self._unpacksizes[i - shift])
        return result