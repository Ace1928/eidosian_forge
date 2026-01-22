import os
import pytest
import textwrap
import numpy as np
from . import util
class TestFixedString(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'string', 'fixed_string.f90')]

    @staticmethod
    def _sint(s, start=0, end=None):
        """Return the content of a string buffer as integer value.

        For example:
          _sint('1234') -> 4321
          _sint('123A') -> 17321
        """
        if isinstance(s, np.ndarray):
            s = s.tobytes()
        elif isinstance(s, str):
            s = s.encode()
        assert isinstance(s, bytes)
        if end is None:
            end = len(s)
        i = 0
        for j in range(start, min(end, len(s))):
            i += s[j] * 10 ** j
        return i

    def _get_input(self, intent='in'):
        if intent in ['in']:
            yield ''
            yield '1'
            yield '1234'
            yield '12345'
            yield b''
            yield b'\x00'
            yield b'1'
            yield b'\x01'
            yield b'1\x00'
            yield b'1234'
            yield b'12345'
        yield np.ndarray((), np.bytes_, buffer=b'')
        yield np.array(b'')
        yield np.array(b'\x00')
        yield np.array(b'1')
        yield np.array(b'1\x00')
        yield np.array(b'\x01')
        yield np.array(b'1234')
        yield np.array(b'123\x00')
        yield np.array(b'12345')

    def test_intent_in(self):
        for s in self._get_input():
            r = self.module.test_in_bytes4(s)
            expected = self._sint(s, end=4)
            assert r == expected, s

    def test_intent_inout(self):
        for s in self._get_input(intent='inout'):
            rest = self._sint(s, start=4)
            r = self.module.test_inout_bytes4(s)
            expected = self._sint(s, end=4)
            assert r == expected
            assert rest == self._sint(s, start=4)