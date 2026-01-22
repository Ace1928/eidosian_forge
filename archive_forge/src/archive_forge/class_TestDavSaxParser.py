import stat
from http.client import parse_headers
from io import StringIO
from breezy import errors, tests
from breezy.plugins.webdav import webdav
from breezy.tests import http_server
class TestDavSaxParser(tests.TestCase):

    def _extract_dir_content_from_str(self, str):
        return webdav._extract_dir_content('http://localhost/blah', StringIO(str))

    def _extract_stat_from_str(self, str):
        return webdav._extract_stat_info('http://localhost/blah', StringIO(str))

    def test_unkown_format_response(self):
        example = '<document/>'
        self.assertRaises(errors.InvalidHttpResponse, self._extract_dir_content_from_str, example)

    def test_list_dir_malformed_response(self):
        example = '<?xml version="1.0" encoding="utf-8"?>\n<D:multistatus xmlns:D="DAV:" xmlns:ns0="urn:uuid:c2f41010-65b3-11d1-a29f-00aa00c14882/">\n<D:response>\n<D:href>http://localhost/</D:href>'
        self.assertRaises(errors.InvalidHttpResponse, self._extract_dir_content_from_str, example)

    def test_list_dir_incomplete_format_response(self):
        example = '<?xml version="1.0" encoding="utf-8"?>\n<D:multistatus xmlns:D="DAV:" xmlns:ns0="urn:uuid:c2f41010-65b3-11d1-a29f-00aa00c14882/">\n<D:response>\n<D:href>http://localhost/</D:href>\n</D:response>\n<D:response>\n<D:href>http://localhost/titi</D:href>\n</D:response>\n<D:href>http://localhost/toto</D:href>\n</D:multistatus>'
        self.assertRaises(errors.NotADirectory, self._extract_dir_content_from_str, example)

    def test_list_dir_apache2_example(self):
        example = _get_list_dir_apache2_depth_1_prop()
        self.assertRaises(errors.NotADirectory, self._extract_dir_content_from_str, example)

    def test_list_dir_lighttpd_example(self):
        example = '<?xml version="1.0" encoding="utf-8"?>\n<D:multistatus xmlns:D="DAV:" xmlns:ns0="urn:uuid:c2f41010-65b3-11d1-a29f-00aa00c14882/">\n<D:response>\n<D:href>http://localhost/</D:href>\n</D:response>\n<D:response>\n<D:href>http://localhost/titi</D:href>\n</D:response>\n<D:response>\n<D:href>http://localhost/toto</D:href>\n</D:response>\n</D:multistatus>'
        self.assertRaises(errors.NotADirectory, self._extract_dir_content_from_str, example)

    def test_list_dir_apache2_dir_depth_1_example(self):
        example = _get_list_dir_apache2_depth_1_allprop()
        self.assertEqual([('executable', False, 14, True), ('read-only', False, 42, False), ('titi', False, 6, False), ('toto', True, -1, False)], self._extract_dir_content_from_str(example))

    def test_stat_malformed_response(self):
        example = '<?xml version="1.0" encoding="utf-8"?>\n<D:multistatus xmlns:D="DAV:" xmlns:ns0="urn:uuid:c2f41010-65b3-11d1-a29f-00aa00c14882/">\n<D:response>\n<D:href>http://localhost/</D:href>'
        self.assertRaises(errors.InvalidHttpResponse, self._extract_stat_from_str, example)

    def test_stat_incomplete_format_response(self):
        example = '<?xml version="1.0" encoding="utf-8"?>\n<D:multistatus xmlns:D="DAV:" xmlns:ns0="urn:uuid:c2f41010-65b3-11d1-a29f-00aa00c14882/">\n<D:href>http://localhost/toto</D:href>\n</D:multistatus>'
        self.assertRaises(errors.InvalidHttpResponse, self._extract_stat_from_str, example)

    def test_stat_apache2_file_example(self):
        example = '<?xml version="1.0" encoding="utf-8"?>\n<D:multistatus xmlns:D="DAV:" xmlns:ns0="DAV:">\n<D:response xmlns:lp1="DAV:" xmlns:lp2="http://apache.org/dav/props/">\n<D:href>/executable</D:href>\n<D:propstat>\n<D:prop>\n<lp1:resourcetype/>\n<lp1:creationdate>2008-06-08T09:50:15Z</lp1:creationdate>\n<lp1:getcontentlength>12</lp1:getcontentlength>\n<lp1:getlastmodified>Sun, 08 Jun 2008 09:50:11 GMT</lp1:getlastmodified>\n<lp1:getetag>"da9f81-0-9ef33ac0"</lp1:getetag>\n<lp2:executable>T</lp2:executable>\n<D:supportedlock>\n<D:lockentry>\n<D:lockscope><D:exclusive/></D:lockscope>\n<D:locktype><D:write/></D:locktype>\n</D:lockentry>\n<D:lockentry>\n<D:lockscope><D:shared/></D:lockscope>\n<D:locktype><D:write/></D:locktype>\n</D:lockentry>\n</D:supportedlock>\n<D:lockdiscovery/>\n</D:prop>\n<D:status>HTTP/1.1 200 OK</D:status>\n</D:propstat>\n</D:response>\n</D:multistatus>'
        st = self._extract_stat_from_str(example)
        self.assertEqual(12, st.st_size)
        self.assertFalse(stat.S_ISDIR(st.st_mode))
        self.assertTrue(stat.S_ISREG(st.st_mode))
        self.assertTrue(st.st_mode & stat.S_IXUSR)

    def test_stat_apache2_dir_depth_1_example(self):
        example = _get_list_dir_apache2_depth_1_allprop()
        self.assertRaises(errors.InvalidHttpResponse, self._extract_stat_from_str, example)

    def test_stat_apache2_dir_depth_0_example(self):
        example = '<?xml version="1.0" encoding="utf-8"?>\n<D:multistatus xmlns:D="DAV:" xmlns:ns0="DAV:">\n<D:response xmlns:lp1="DAV:" xmlns:lp2="http://apache.org/dav/props/">\n<D:href>/</D:href>\n<D:propstat>\n<D:prop>\n<lp1:resourcetype><D:collection/></lp1:resourcetype>\n<lp1:creationdate>2008-06-08T10:50:38Z</lp1:creationdate>\n<lp1:getlastmodified>Sun, 08 Jun 2008 10:50:38 GMT</lp1:getlastmodified>\n<lp1:getetag>"da7f5a-cc-7722db80"</lp1:getetag>\n<D:supportedlock>\n<D:lockentry>\n<D:lockscope><D:exclusive/></D:lockscope>\n<D:locktype><D:write/></D:locktype>\n</D:lockentry>\n<D:lockentry>\n<D:lockscope><D:shared/></D:lockscope>\n<D:locktype><D:write/></D:locktype>\n</D:lockentry>\n</D:supportedlock>\n<D:lockdiscovery/>\n</D:prop>\n<D:status>HTTP/1.1 200 OK</D:status>\n</D:propstat>\n</D:response>\n</D:multistatus>\n'
        st = self._extract_stat_from_str(example)
        self.assertEqual(-1, st.st_size)
        self.assertTrue(stat.S_ISDIR(st.st_mode))
        self.assertTrue(st.st_mode & stat.S_IXUSR)