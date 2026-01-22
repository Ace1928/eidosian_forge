import sys
import time
from uuid import UUID
import pytest
from cherrypy._cpcompat import text_or_bytes
def _read_marked_region(self, marker=None):
    """Return lines from self.logfile in the marked region.

        If marker is None, self.lastmarker is used. If the log hasn't
        been marked (using self.markLog), the entire log will be returned.
        """
    logfile = self.logfile
    marker = marker or self.lastmarker
    if marker is None:
        with open(logfile, 'rb') as f:
            return f.readlines()
    if isinstance(marker, str):
        marker = marker.encode('utf-8')
    data = []
    in_region = False
    with open(logfile, 'rb') as f:
        for line in f:
            if in_region:
                if line.startswith(self.markerPrefix) and marker not in line:
                    break
                else:
                    data.append(line)
            elif marker in line:
                in_region = True
    return data