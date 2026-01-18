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
def test_post_multipart(self):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    contents = ''.join([c * 65536 for c in alphabet])
    files = [('file', 'file.txt', contents)]
    content_type, body = encode_multipart_formdata(files)
    body = body.encode('Latin-1')
    c = self.make_connection()
    c.putrequest('POST', '/post_multipart')
    c.putheader('Content-Type', content_type)
    c.putheader('Content-Length', str(len(body)))
    c.endheaders()
    c.send(body)
    response = c.getresponse()
    self.body = response.fp.read()
    self.status = str(response.status)
    self.assertStatus(200)
    parts = ['%s * 65536' % ch for ch in alphabet]
    self.assertBody(', '.join(parts))
    response.close()
    c.close()