import datetime
import io
import logging
import os
import re
import subprocess
import sys
import time
import unittest
import warnings
import contextlib
import portend
import pytest
from cheroot.test import webtest
import cherrypy
from cherrypy._cpcompat import text_or_bytes, HTTPSConnection, ntob
from cherrypy.lib import httputil
from cherrypy.lib import gctools
class NativeServerSupervisor(LocalSupervisor):
    """Server supervisor for the builtin HTTP server."""
    httpserver_class = 'cherrypy._cpnative_server.CPHTTPServer'
    using_apache = False
    using_wsgi = False

    def __str__(self):
        return 'Builtin HTTP Server on %s:%s' % (self.host, self.port)