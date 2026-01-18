import logging
import os
import pathlib
import sys
import time
import pytest
def test_uncoloredlogrecord_without_markup():
    from kivy.logger import UncoloredLogRecord
    originallogrecord = logging.LogRecord(name='kivy.test', level=logging.DEBUG, pathname='test.py', lineno=1, msg='Part1: Part2 Part 3', args=('args',), exc_info=None, func='test_colon_splitting', sinfo=None)
    shimmedlogrecord = UncoloredLogRecord(originallogrecord)
    assert str(originallogrecord) == str(shimmedlogrecord)
    originallogrecord = logging.LogRecord(name='kivy.test', level=logging.DEBUG, pathname='test.py', lineno=1, msg=1, args=None, exc_info=None, func='test_colon_splitting', sinfo=None)
    shimmedlogrecord = UncoloredLogRecord(originallogrecord)
    assert str(originallogrecord) == str(shimmedlogrecord)