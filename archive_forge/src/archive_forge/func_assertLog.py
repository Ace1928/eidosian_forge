import sys
import time
from uuid import UUID
import pytest
from cherrypy._cpcompat import text_or_bytes
def assertLog(self, sliceargs, lines, marker=None):
    """Fail if log.readlines()[sliceargs] is not contained in 'lines'.

        The log will be searched from the given marker to the next marker.
        If marker is None, self.lastmarker is used. If the log hasn't
        been marked (using self.markLog), the entire log will be searched.
        """
    data = self._read_marked_region(marker)
    if isinstance(sliceargs, int):
        if isinstance(lines, (tuple, list)):
            lines = lines[0]
        if isinstance(lines, str):
            lines = lines.encode('utf-8')
        if lines not in data[sliceargs]:
            msg = '%r not found on log line %r' % (lines, sliceargs)
            self._handleLogError(msg, [data[sliceargs], '--EXTRA CONTEXT--'] + data[sliceargs + 1:sliceargs + 6], marker, lines)
    else:
        if isinstance(lines, tuple):
            lines = list(lines)
        elif isinstance(lines, text_or_bytes):
            raise TypeError("The 'lines' arg must be a list when 'sliceargs' is a tuple.")
        start, stop = sliceargs
        for line, logline in zip(lines, data[start:stop]):
            if isinstance(line, str):
                line = line.encode('utf-8')
            if line not in logline:
                msg = '%r not found in log' % line
                self._handleLogError(msg, data[start:stop], marker, line)