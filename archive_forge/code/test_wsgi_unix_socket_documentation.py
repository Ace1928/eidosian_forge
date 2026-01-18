import os
import socket
import atexit
import tempfile
from http.client import HTTPConnection
import pytest
import cherrypy
from cherrypy.test import helper

        Override the connect method and assign a unix socket as a transport.
        