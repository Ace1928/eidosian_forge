import stat
from http.client import parse_headers
from io import StringIO
from breezy import errors, tests
from breezy.plugins.webdav import webdav
from breezy.tests import http_server
def test_stat_apache2_dir_depth_0_example(self):
    example = '<?xml version="1.0" encoding="utf-8"?>\n<D:multistatus xmlns:D="DAV:" xmlns:ns0="DAV:">\n<D:response xmlns:lp1="DAV:" xmlns:lp2="http://apache.org/dav/props/">\n<D:href>/</D:href>\n<D:propstat>\n<D:prop>\n<lp1:resourcetype><D:collection/></lp1:resourcetype>\n<lp1:creationdate>2008-06-08T10:50:38Z</lp1:creationdate>\n<lp1:getlastmodified>Sun, 08 Jun 2008 10:50:38 GMT</lp1:getlastmodified>\n<lp1:getetag>"da7f5a-cc-7722db80"</lp1:getetag>\n<D:supportedlock>\n<D:lockentry>\n<D:lockscope><D:exclusive/></D:lockscope>\n<D:locktype><D:write/></D:locktype>\n</D:lockentry>\n<D:lockentry>\n<D:lockscope><D:shared/></D:lockscope>\n<D:locktype><D:write/></D:locktype>\n</D:lockentry>\n</D:supportedlock>\n<D:lockdiscovery/>\n</D:prop>\n<D:status>HTTP/1.1 200 OK</D:status>\n</D:propstat>\n</D:response>\n</D:multistatus>\n'
    st = self._extract_stat_from_str(example)
    self.assertEqual(-1, st.st_size)
    self.assertTrue(stat.S_ISDIR(st.st_mode))
    self.assertTrue(st.st_mode & stat.S_IXUSR)