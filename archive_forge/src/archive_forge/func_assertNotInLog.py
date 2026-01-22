import sys
import time
from uuid import UUID
import pytest
from cherrypy._cpcompat import text_or_bytes
def assertNotInLog(self, line, marker=None):
    """Fail if the given (partial) line is in the log.

        The log will be searched from the given marker to the next marker.
        If marker is None, self.lastmarker is used. If the log hasn't
        been marked (using self.markLog), the entire log will be searched.
        """
    data = self._read_marked_region(marker)
    for logline in data:
        if line in logline:
            msg = '%r found in log' % line
            self._handleLogError(msg, data, marker, line)