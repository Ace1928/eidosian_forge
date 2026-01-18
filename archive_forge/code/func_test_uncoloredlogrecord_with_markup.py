import logging
import os
import pathlib
import sys
import time
import pytest
def test_uncoloredlogrecord_with_markup():
    from kivy.logger import UncoloredLogRecord
    originallogrecord = logging.LogRecord(name='kivy.test', level=logging.DEBUG, pathname='test.py', lineno=1, msg='Part1: $BOLDPart2$RESET Part 3', args=('args',), exc_info=None, func='test_colon_splitting', sinfo=None)
    shimmedlogrecord = UncoloredLogRecord(originallogrecord)
    assert str(shimmedlogrecord) == '<LogRecord: kivy.test, 10, test.py, 1, "Part1: Part2 Part 3">'