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
def _setup_mimetypes():
    """Pre-initialize global mimetype map."""
    if not mimetypes.inited:
        mimetypes.init()
    mimetypes.types_map['.dwg'] = 'image/x-dwg'
    mimetypes.types_map['.ico'] = 'image/x-icon'
    mimetypes.types_map['.bz2'] = 'application/x-bzip2'
    mimetypes.types_map['.gz'] = 'application/x-gzip'