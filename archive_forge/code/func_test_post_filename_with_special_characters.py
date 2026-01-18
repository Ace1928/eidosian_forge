import errno
import mimetypes
import socket
import sys
from unittest import mock
import urllib.parse
from http.client import HTTPConnection
import cherrypy
from cherrypy._cpcompat import HTTPSConnection
from cherrypy.test import helper
def test_post_filename_with_special_characters(self):
    """Testing that we can handle filenames with special characters.

        This was reported as a bug in:

        * https://github.com/cherrypy/cherrypy/issues/1146/
        * https://github.com/cherrypy/cherrypy/issues/1397/
        * https://github.com/cherrypy/cherrypy/issues/1694/
        """
    fnames = ['boop.csv', 'foo, bar.csv', 'bar, xxxx.csv', 'file"name.csv', 'file;name.csv', 'file; name.csv', u'test_łóąä.txt']
    for fname in fnames:
        files = [('myfile', fname, 'yunyeenyunyue')]
        content_type, body = encode_multipart_formdata(files)
        body = body.encode('Latin-1')
        c = self.make_connection()
        c.putrequest('POST', '/post_filename')
        c.putheader('Content-Type', content_type)
        c.putheader('Content-Length', str(len(body)))
        c.endheaders()
        c.send(body)
        response = c.getresponse()
        self.body = response.fp.read()
        self.status = str(response.status)
        self.assertStatus(200)
        self.assertBody(fname)
        response.close()
        c.close()