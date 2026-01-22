import csv
import json
import logging
class _LineGenerator(object):
    """A csv line generator that allows feeding lines to a csv.DictReader."""

    def __init__(self):
        self._lines = []

    def push_line(self, line):
        assert not self._lines
        self._lines.append(line)

    def __iter__(self):
        return self

    def next(self):
        line_length = len(self._lines)
        if line_length == 0:
            raise DecodeError('Columns do not match specified csv headers: empty line was found')
        assert line_length == 1, 'Unexpected number of lines %s' % line_length
        return self._lines.pop()