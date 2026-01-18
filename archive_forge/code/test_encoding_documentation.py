import gzip
import io
from unittest import mock
from http.client import IncompleteRead
from urllib.parse import quote as url_quote
import cherrypy
from cherrypy._cpcompat import ntob, ntou
from cherrypy.test import helper

        Decoding of a multipart entity should also pass when
        the entity is bigger than maxrambytes. See ticket #1352.
        