import sys
import time
from uuid import UUID
import pytest
from cherrypy._cpcompat import text_or_bytes
def assertValidUUIDv4(self, marker=None):
    """Fail if the given UUIDv4 is not valid.

        The log will be searched from the given marker to the next marker.
        If marker is None, self.lastmarker is used. If the log hasn't
        been marked (using self.markLog), the entire log will be searched.
        """
    data = self._read_marked_region(marker)
    data = [chunk.decode('utf-8').rstrip('\n').rstrip('\r') for chunk in data]
    for log_chunk in data:
        try:
            uuid_log = data[-1]
            uuid_obj = UUID(uuid_log, version=4)
        except (TypeError, ValueError):
            pass
        else:
            if str(uuid_obj) == uuid_log:
                return
            msg = '%r is not a valid UUIDv4' % uuid_log
            self._handleLogError(msg, data, marker, log_chunk)
    msg = 'UUIDv4 not found in log'
    self._handleLogError(msg, data, marker, log_chunk)