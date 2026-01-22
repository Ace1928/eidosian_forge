import io
import socket
import sys
import threading
from http.client import UnknownProtocol, parse_headers
from http.server import SimpleHTTPRequestHandler
import breezy
from .. import (config, controldir, debug, errors, osutils, tests, trace,
from ..bzr import remote as _mod_remote
from ..transport import remote
from ..transport.http import urllib
from ..transport.http.urllib import (AbstractAuthHandler, BasicAuthHandler,
from . import features, http_server, http_utils, test_server
from .scenarios import load_tests_apply_scenarios, multiply_scenarios
class TruncatedBeforeBoundaryRequestHandler(http_server.TestingHTTPRequestHandler):
    """Truncation before a boundary, like in bug 198646"""
    _truncated_ranges = 1

    def get_multiple_ranges(self, file, file_size, ranges):
        self.send_response(206)
        self.send_header('Accept-Ranges', 'bytes')
        boundary = 'tagada'
        self.send_header('Content-Type', 'multipart/byteranges; boundary=%s' % boundary)
        boundary_line = b'--%s\r\n' % boundary.encode('ascii')
        content_length = 0
        for start, end in ranges:
            content_length += len(boundary_line)
            content_length += self._header_line_length('Content-type', 'application/octet-stream')
            content_length += self._header_line_length('Content-Range', 'bytes %d-%d/%d' % (start, end, file_size))
            content_length += len('\r\n')
            content_length += end - start
        content_length += len(boundary_line)
        self.send_header('Content-length', content_length)
        self.end_headers()
        cur = 0
        for start, end in ranges:
            if cur + self._truncated_ranges >= len(ranges):
                self.close_connection = 1
                return
            self.wfile.write(boundary_line)
            self.send_header('Content-type', 'application/octet-stream')
            self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, end, file_size))
            self.end_headers()
            self.send_range_content(file, start, end - start + 1)
            cur += 1
        self.wfile.write(boundary_line)