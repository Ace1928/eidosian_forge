import errno
import os
import random
import re
import socket
import time
from hashlib import md5
import six.moves.http_client as httplib
from six.moves import urllib as urlparse
from boto import config, UserAgent
from boto.connection import AWSAuthConnection
from boto.exception import InvalidUriError
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableUploadException
from boto.s3.keyfile import KeyFile
def _query_server_pos(self, conn, file_length):
    """
        Queries server to find out what bytes it currently has.

        Returns (server_start, server_end), where the values are inclusive.
        For example, (0, 2) would mean that the server has bytes 0, 1, *and* 2.

        Raises ResumableUploadException if problem querying server.
        """
    resp = self._query_server_state(conn, file_length)
    if resp.status == 200:
        return (0, file_length - 1)
    if resp.status != 308:
        raise ResumableUploadException('Got non-308 response (%s) from server state query' % resp.status, ResumableTransferDisposition.START_OVER)
    got_valid_response = False
    range_spec = resp.getheader('range')
    if range_spec:
        m = re.search('bytes=(\\d+)-(\\d+)', range_spec)
        if m:
            server_start = long(m.group(1))
            server_end = long(m.group(2))
            got_valid_response = True
    else:
        return self.SERVER_HAS_NOTHING
    if not got_valid_response:
        raise ResumableUploadException("Couldn't parse upload server state query response (%s)" % str(resp.getheaders()), ResumableTransferDisposition.START_OVER)
    if conn.debug >= 1:
        print('Server has: Range: %d - %d.' % (server_start, server_end))
    return (server_start, server_end)