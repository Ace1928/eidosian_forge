from __future__ import absolute_import, division, print_function
import os
import re
from contextlib import contextmanager
from struct import Struct
from ansible.module_utils.six import PY3
class OpensshParser(object):
    """Parser for OpenSSH encoded objects"""
    BOOLEAN_OFFSET = 1
    UINT32_OFFSET = 4
    UINT64_OFFSET = 8

    def __init__(self, data):
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError('Data must be bytes-like not %s' % type(data))
        self._data = memoryview(data) if PY3 else data
        self._pos = 0

    def boolean(self):
        next_pos = self._check_position(self.BOOLEAN_OFFSET)
        value = _BOOLEAN.unpack(self._data[self._pos:next_pos])[0]
        self._pos = next_pos
        return value

    def uint32(self):
        next_pos = self._check_position(self.UINT32_OFFSET)
        value = _UINT32.unpack(self._data[self._pos:next_pos])[0]
        self._pos = next_pos
        return value

    def uint64(self):
        next_pos = self._check_position(self.UINT64_OFFSET)
        value = _UINT64.unpack(self._data[self._pos:next_pos])[0]
        self._pos = next_pos
        return value

    def string(self):
        length = self.uint32()
        next_pos = self._check_position(length)
        value = self._data[self._pos:next_pos]
        self._pos = next_pos
        return value if not PY3 else bytes(value)

    def mpint(self):
        return self._big_int(self.string(), 'big', signed=True)

    def name_list(self):
        raw_string = self.string()
        return raw_string.decode('ASCII').split(',')

    def string_list(self):
        result = []
        raw_string = self.string()
        if raw_string:
            parser = OpensshParser(raw_string)
            while parser.remaining_bytes():
                result.append(parser.string())
        return result

    def option_list(self):
        result = []
        raw_string = self.string()
        if raw_string:
            parser = OpensshParser(raw_string)
            while parser.remaining_bytes():
                name = parser.string()
                data = parser.string()
                if data:
                    data = OpensshParser(data).string()
                result.append((name, data))
        return result

    def seek(self, offset):
        self._pos = self._check_position(offset)
        return self._pos

    def remaining_bytes(self):
        return len(self._data) - self._pos

    def _check_position(self, offset):
        if self._pos + offset > len(self._data):
            raise ValueError('Insufficient data remaining at position: %s' % self._pos)
        elif self._pos + offset < 0:
            raise ValueError('Position cannot be less than zero.')
        else:
            return self._pos + offset

    @classmethod
    def signature_data(cls, signature_string):
        signature_data = {}
        parser = cls(signature_string)
        signature_type = parser.string()
        signature_blob = parser.string()
        blob_parser = cls(signature_blob)
        if signature_type in (b'ssh-rsa', b'rsa-sha2-256', b'rsa-sha2-512'):
            signature_data['s'] = cls._big_int(signature_blob, 'big')
        elif signature_type == b'ssh-dss':
            signature_data['r'] = cls._big_int(signature_blob[:20], 'big')
            signature_data['s'] = cls._big_int(signature_blob[20:], 'big')
        elif signature_type in (b'ecdsa-sha2-nistp256', b'ecdsa-sha2-nistp384', b'ecdsa-sha2-nistp521'):
            signature_data['r'] = blob_parser.mpint()
            signature_data['s'] = blob_parser.mpint()
        elif signature_type == b'ssh-ed25519':
            signature_data['R'] = cls._big_int(signature_blob[:32], 'little')
            signature_data['S'] = cls._big_int(signature_blob[32:], 'little')
        else:
            raise ValueError('%s is not a valid signature type' % signature_type)
        signature_data['signature_type'] = signature_type
        return signature_data

    @classmethod
    def _big_int(cls, raw_string, byte_order, signed=False):
        if byte_order not in ('big', 'little'):
            raise ValueError('Byte_order must be one of (big, little) not %s' % byte_order)
        if PY3:
            return int.from_bytes(raw_string, byte_order, signed=signed)
        result = 0
        byte_length = len(raw_string)
        if byte_length > 0:
            msb = raw_string[0] if byte_order == 'big' else raw_string[-1]
            negative = bool(ord(msb) & 128)
            pad = b'\xff' if signed and negative else b'\x00'
            pad_length = 4 - byte_length % 4
            if pad_length < 4:
                raw_string = pad * pad_length + raw_string if byte_order == 'big' else raw_string + pad * pad_length
                byte_length += pad_length
            if byte_order == 'big':
                for i in range(0, byte_length, cls.UINT32_OFFSET):
                    left_shift = result << cls.UINT32_OFFSET * 8
                    result = left_shift + _UINT32.unpack(raw_string[i:i + cls.UINT32_OFFSET])[0]
            else:
                for i in range(byte_length, 0, -cls.UINT32_OFFSET):
                    left_shift = result << cls.UINT32_OFFSET * 8
                    result = left_shift + _UINT32_LE.unpack(raw_string[i - cls.UINT32_OFFSET:i])[0]
            if signed and negative:
                result -= 1 << 8 * byte_length
        return result