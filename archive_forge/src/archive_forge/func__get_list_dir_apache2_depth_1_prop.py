import stat
from http.client import parse_headers
from io import StringIO
from breezy import errors, tests
from breezy.plugins.webdav import webdav
from breezy.tests import http_server
def _get_list_dir_apache2_depth_1_prop():
    return '<?xml version="1.0" encoding="utf-8"?>\n<D:multistatus xmlns:D="DAV:" xmlns:ns0="DAV:">\n    <D:response>\n        <D:href>/19016477731212686926.835527/</D:href>\n        <D:propstat>\n            <D:prop>\n            </D:prop>\n            <D:status>HTTP/1.1 200 OK</D:status>\n        </D:propstat>\n    </D:response>\n    <D:response>\n        <D:href>/19016477731212686926.835527/a</D:href>\n        <D:propstat>\n            <D:prop>\n            </D:prop>\n            <D:status>HTTP/1.1 200 OK</D:status>\n        </D:propstat>\n    </D:response>\n    <D:response>\n        <D:href>/19016477731212686926.835527/b</D:href>\n        <D:propstat>\n            <D:prop>\n            </D:prop>\n            <D:status>HTTP/1.1 200 OK</D:status>\n        </D:propstat>\n    </D:response>\n    <D:response>\n        <D:href>/19016477731212686926.835527/c/</D:href>\n        <D:propstat>\n            <D:prop>\n            </D:prop>\n            <D:status>HTTP/1.1 200 OK</D:status>\n        </D:propstat>\n    </D:response>\n</D:multistatus>'