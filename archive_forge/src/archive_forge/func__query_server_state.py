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
def _query_server_state(self, conn, file_length):
    """
        Queries server to find out state of given upload.

        Note that this method really just makes special case use of the
        fact that the upload server always returns the current start/end
        state whenever a PUT doesn't complete.

        Returns HTTP response from sending request.

        Raises ResumableUploadException if problem querying server.
        """
    put_headers = {}
    put_headers['Content-Range'] = self._build_content_range_header('*', file_length)
    put_headers['Content-Length'] = '0'
    return AWSAuthConnection.make_request(conn, 'PUT', path=self.tracker_uri_path, auth_path=self.tracker_uri_path, headers=put_headers, host=self.tracker_uri_host)