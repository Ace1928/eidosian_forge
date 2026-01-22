import mimetypes
import os
import platform
import re
import stat
import unicodedata
import urllib.parse
from email.generator import _make_boundary as make_boundary
from io import UnsupportedOperation
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.lib import cptools, file_generator_limited, httputil
def _serve_fileobj(fileobj, content_type, content_length, debug=False):
    """Set ``response.body`` to the given file object, perhaps ranged.

    Internal helper.
    """
    response = cherrypy.serving.response
    request = cherrypy.serving.request
    if request.protocol >= (1, 1):
        response.headers['Accept-Ranges'] = 'bytes'
        r = httputil.get_ranges(request.headers.get('Range'), content_length)
        if r == []:
            response.headers['Content-Range'] = 'bytes */%s' % content_length
            message = 'Invalid Range (first-byte-pos greater than Content-Length)'
            if debug:
                cherrypy.log(message, 'TOOLS.STATIC')
            raise cherrypy.HTTPError(416, message)
        if r:
            if len(r) == 1:
                start, stop = r[0]
                if stop > content_length:
                    stop = content_length
                r_len = stop - start
                if debug:
                    cherrypy.log('Single part; start: %r, stop: %r' % (start, stop), 'TOOLS.STATIC')
                response.status = '206 Partial Content'
                response.headers['Content-Range'] = 'bytes %s-%s/%s' % (start, stop - 1, content_length)
                response.headers['Content-Length'] = r_len
                fileobj.seek(start)
                response.body = file_generator_limited(fileobj, r_len)
            else:
                response.status = '206 Partial Content'
                boundary = make_boundary()
                ct = 'multipart/byteranges; boundary=%s' % boundary
                response.headers['Content-Type'] = ct
                if 'Content-Length' in response.headers:
                    del response.headers['Content-Length']

                def file_ranges():
                    yield b'\r\n'
                    for start, stop in r:
                        if debug:
                            cherrypy.log('Multipart; start: %r, stop: %r' % (start, stop), 'TOOLS.STATIC')
                        yield ntob('--' + boundary, 'ascii')
                        yield ntob('\r\nContent-type: %s' % content_type, 'ascii')
                        yield ntob('\r\nContent-range: bytes %s-%s/%s\r\n\r\n' % (start, stop - 1, content_length), 'ascii')
                        fileobj.seek(start)
                        gen = file_generator_limited(fileobj, stop - start)
                        for chunk in gen:
                            yield chunk
                        yield b'\r\n'
                    yield ntob('--' + boundary + '--', 'ascii')
                    yield b'\r\n'
                response.body = file_ranges()
            return response.body
        elif debug:
            cherrypy.log('No byteranges requested', 'TOOLS.STATIC')
    response.headers['Content-Length'] = content_length
    response.body = fileobj
    return response.body