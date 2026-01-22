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
class LocalWSGISupervisor(LocalSupervisor):
    """Server supervisor for the builtin WSGI server."""
    httpserver_class = 'cherrypy._cpwsgi_server.CPWSGIServer'
    using_apache = False
    using_wsgi = True

    def __str__(self):
        return 'Builtin WSGI Server on %s:%s' % (self.host, self.port)

    def sync_apps(self):
        """Hook a new WSGI app into the origin server."""
        cherrypy.server.httpserver.wsgi_app = self.get_app()

    def get_app(self, app=None):
        """Obtain a new (decorated) WSGI app to hook into the origin server."""
        if app is None:
            app = cherrypy.tree
        if self.validate:
            try:
                from wsgiref import validate
            except ImportError:
                warnings.warn('Error importing wsgiref. The validator will not run.')
            else:
                app = validate.validator(app)
        return app