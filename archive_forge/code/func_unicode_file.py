import io
import os
import sys
import re
import platform
import tempfile
import urllib.parse
import unittest.mock
from http.client import HTTPConnection
import pytest
import py.path
import path
import cherrypy
from cherrypy.lib import static
from cherrypy._cpcompat import HTTPSConnection, ntou, tonative
from cherrypy.test import helper
@classmethod
def unicode_file(cls):
    filename = ntou('Слава Україні.html', 'utf-8')
    filepath = curdir / 'static' / filename
    with filepath.open('w', encoding='utf-8') as strm:
        strm.write(ntou('Героям Слава!', 'utf-8'))
    cls.files_to_remove.append(filepath)