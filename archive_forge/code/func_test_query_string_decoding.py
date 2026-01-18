import gzip
import io
from unittest import mock
from http.client import IncompleteRead
from urllib.parse import quote as url_quote
import cherrypy
from cherrypy._cpcompat import ntob, ntou
from cherrypy.test import helper
def test_query_string_decoding(self):
    URI_TMPL = '/reqparams?q={q}'
    europoundUtf8_2_bytes = europoundUnicode.encode('utf-8')
    europoundUtf8_2nd_byte = europoundUtf8_2_bytes[1:2]
    self.getPage(URI_TMPL.format(q=url_quote(europoundUtf8_2_bytes)))
    self.assertBody(b'q: ' + europoundUtf8_2_bytes)
    self.getPage(URI_TMPL.format(q=url_quote(europoundUtf8_2nd_byte)))
    self.assertStatus(404)
    self.assertErrorPage(404, "The given query string could not be processed. Query strings for this resource must be encoded with 'utf8'.")