from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
class LineBasedParser(object):

    def __init__(self, input_stream):
        """A Parser that keeps track of line numbers.

        :param input: the file-like object to read from
        """
        self.input = input_stream
        self.lineno = 0
        self._buffer = []

    def abort(self, exception, *args):
        """Raise an exception providing line number information."""
        raise exception(self.lineno, *args)

    def readline(self):
        """Get the next line including the newline or '' on EOF."""
        self.lineno += 1
        if self._buffer:
            return self._buffer.pop()
        else:
            return self.input.readline()

    def next_line(self):
        """Get the next line without the newline or None on EOF."""
        line = self.readline()
        if line:
            return line[:-1]
        else:
            return None

    def push_line(self, line):
        """Push line back onto the line buffer.

        :param line: the line with no trailing newline
        """
        self.lineno -= 1
        self._buffer.append(line + b'\n')

    def read_bytes(self, count):
        """Read a given number of bytes from the input stream.

        Throws MissingBytes if the bytes are not found.

        Note: This method does not read from the line buffer.

        :return: a string
        """
        result = self.input.read(count)
        found = len(result)
        self.lineno += result.count(b'\n')
        if found != count:
            self.abort(errors.MissingBytes, count, found)
        return result

    def read_until(self, terminator):
        """Read the input stream until the terminator is found.

        Throws MissingTerminator if the terminator is not found.

        Note: This method does not read from the line buffer.

        :return: the bytes read up to but excluding the terminator.
        """
        lines = []
        term = terminator + b'\n'
        while True:
            line = self.input.readline()
            if line == term:
                break
            else:
                lines.append(line)
        return b''.join(lines)