import stat
from http.client import parse_headers
from io import StringIO
from breezy import errors, tests
from breezy.plugins.webdav import webdav
from breezy.tests import http_server
def _extract_stat_from_str(self, str):
    return webdav._extract_stat_info('http://localhost/blah', StringIO(str))