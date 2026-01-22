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
class SupportedMethods:
    """Hold list of methods."""
    formats: List[Dict[str, Any]] = [{'name': '7z', 'magic': MAGIC_7Z}]
    methods: List[Dict[str, Any]] = [{'id': COMPRESSION_METHOD.COPY, 'name': 'COPY', 'native': False, 'need_prop': False, 'filter_id': FILTER_COPY, 'type': MethodsType.compressor}, {'id': COMPRESSION_METHOD.LZMA2, 'name': 'LZMA2', 'native': True, 'need_prop': True, 'filter_id': FILTER_LZMA2, 'type': MethodsType.compressor}, {'id': COMPRESSION_METHOD.DELTA, 'name': 'DELTA', 'native': True, 'need_prop': True, 'filter_id': FILTER_DELTA, 'type': MethodsType.filter}, {'id': COMPRESSION_METHOD.LZMA, 'name': 'LZMA', 'native': True, 'need_prop': True, 'filter_id': FILTER_LZMA, 'type': MethodsType.compressor}, {'id': COMPRESSION_METHOD.P7Z_BCJ, 'name': 'BCJ', 'native': True, 'need_prop': False, 'filter_id': FILTER_X86, 'type': MethodsType.filter}, {'id': COMPRESSION_METHOD.BCJ_PPC, 'name': 'PPC', 'native': True, 'need_prop': False, 'filter_id': FILTER_POWERPC, 'type': MethodsType.filter}, {'id': COMPRESSION_METHOD.BCJ_IA64, 'name': 'IA64', 'native': True, 'need_prop': False, 'filter_id': FILTER_IA64, 'type': MethodsType.filter}, {'id': COMPRESSION_METHOD.BCJ_ARM, 'name': 'ARM', 'native': True, 'need_prop': False, 'filter_id': FILTER_ARM, 'type': MethodsType.filter}, {'id': COMPRESSION_METHOD.BCJ_ARMT, 'name': 'ARMT', 'native': True, 'need_prop': False, 'filter_id': FILTER_ARMTHUMB, 'type': MethodsType.filter}, {'id': COMPRESSION_METHOD.BCJ_SPARC, 'name': 'SPARC', 'native': True, 'need_prop': False, 'filter_id': FILTER_SPARC, 'type': MethodsType.filter}, {'id': COMPRESSION_METHOD.MISC_DEFLATE, 'name': 'DEFLATE', 'native': False, 'need_prop': False, 'filter_id': FILTER_DEFLATE, 'type': MethodsType.compressor}, {'id': COMPRESSION_METHOD.MISC_BZIP2, 'name': 'BZip2', 'native': False, 'need_prop': False, 'filter_id': FILTER_BZIP2, 'type': MethodsType.compressor}, {'id': COMPRESSION_METHOD.MISC_ZSTD, 'name': 'ZStandard', 'native': False, 'need_prop': True, 'filter_id': FILTER_ZSTD, 'type': MethodsType.compressor}, {'id': COMPRESSION_METHOD.PPMD, 'name': 'PPMd', 'native': False, 'need_prop': True, 'filter_id': FILTER_PPMD, 'type': MethodsType.compressor}, {'id': COMPRESSION_METHOD.MISC_BROTLI, 'name': 'Brotli', 'native': False, 'need_prop': True, 'filter_id': FILTER_BROTLI, 'type': MethodsType.compressor}, {'id': COMPRESSION_METHOD.MISC_DEFLATE64, 'name': 'DEFLATE64', 'native': False, 'need_prop': False, 'filter_id': FILTER_DEFLATE64, 'type': MethodsType.compressor}, {'id': COMPRESSION_METHOD.CRYPT_AES256_SHA256, 'name': '7zAES', 'native': False, 'need_prop': True, 'filter_id': FILTER_CRYPTO_AES256_SHA256, 'type': MethodsType.crypto}]

    @classmethod
    def _find_method(cls, key_id, key_value):
        return next((item for item in cls.methods if item[key_id] == key_value), None)

    @classmethod
    def get_method_name_id(cls, filter_id):
        method = cls._find_method('filter_id', filter_id)
        return method['name']

    @classmethod
    def get_filter_id(cls, coder):
        method = cls._find_method('id', coder['method'])
        if method is None:
            return None
        return method['filter_id']

    @classmethod
    def is_native_filter(cls, filter) -> bool:
        method = cls._find_method('filter_id', filter['id'])
        if method is None:
            raise UnsupportedCompressionMethodError(filter['id'], 'Unknown method id is given.')
        return method['native']

    @classmethod
    def is_compressor(cls, filter):
        method = cls._find_method('filter_id', filter['id'])
        return method['type'] == MethodsType.compressor

    @classmethod
    def is_compressor_id(cls, filter_id):
        method = cls._find_method('filter_id', filter_id)
        return method['type'] == MethodsType.compressor

    @classmethod
    def is_native_coder(cls, coder) -> bool:
        method = cls._find_method('id', coder['method'])
        if method is None:
            cls.raise_unsupported_method_id(coder)
        return method['native']

    @classmethod
    def need_property(cls, filter_id):
        method = cls._find_method('filter_id', filter_id)
        if method is None:
            raise UnsupportedCompressionMethodError(filter_id, 'Found an sunpported filter id.')
        return method['need_prop']

    @classmethod
    def is_crypto_id(cls, filter_id) -> bool:
        method = cls._find_method('filter_id', filter_id)
        if method is None:
            cls.raise_unsupported_filter_id(filter_id)
        return method['type'] == MethodsType.crypto

    @classmethod
    def get_method_id(cls, filter_id) -> bytes:
        method = cls._find_method('filter_id', filter_id)
        if method is None:
            cls.raise_unsupported_filter_id(filter_id)
        return method['id']

    @classmethod
    def get_coder(cls, filter) -> Dict[str, Any]:
        method = cls.get_method_id(filter['id'])
        if filter['id'] in [lzma.FILTER_LZMA1, lzma.FILTER_LZMA2, lzma.FILTER_DELTA]:
            properties: Optional[bytes] = lzma._encode_filter_properties(filter)
        else:
            properties = None
        return {'method': method, 'properties': properties, 'numinstreams': 1, 'numoutstreams': 1}

    @classmethod
    def needs_password(cls, coders) -> bool:
        for coder in coders:
            filter_id = SupportedMethods.get_filter_id(coder)
            if filter_id is None:
                continue
            if SupportedMethods.is_crypto_id(filter_id):
                return True
        return False

    @classmethod
    def raise_unsupported_method_id(cls, coder):
        if coder['method'] == COMPRESSION_METHOD.P7Z_BCJ2:
            raise UnsupportedCompressionMethodError(coder['method'], 'BCJ2 filter is not supported by py7zr. Please consider to contribute to XZ/liblzma project and help Python core team implementing it. Or please use another tool to extract it.')
        if coder['method'] == COMPRESSION_METHOD.MISC_LZ4:
            raise UnsupportedCompressionMethodError(coder['method'], 'Archive is compressed by an unsupported algorythm LZ4.')
        raise UnsupportedCompressionMethodError(coder['method'], 'Archive is compressed by an unsupported compression algorythm.')

    @classmethod
    def raise_unsupported_filter_id(cls, filter_id):
        raise UnsupportedCompressionMethodError(filter_id, 'Found an unsupported filter id is specified.Please use another compression method.')