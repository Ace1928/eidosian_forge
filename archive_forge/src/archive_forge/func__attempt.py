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
def _attempt(filename, content_types, debug=False):
    if debug:
        cherrypy.log('Attempting %r (content_types %r)' % (filename, content_types), 'TOOLS.STATICDIR')
    try:
        content_type = None
        if content_types:
            r, ext = os.path.splitext(filename)
            content_type = content_types.get(ext[1:], None)
        serve_file(filename, content_type=content_type, debug=debug)
        return True
    except cherrypy.NotFound:
        if debug:
            cherrypy.log('NotFound', 'TOOLS.STATICFILE')
        return False