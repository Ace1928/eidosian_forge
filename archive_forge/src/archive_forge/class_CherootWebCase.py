import datetime
import logging
import os
import sys
import time
import threading
import types
import http.client
import cheroot.server
import cheroot.wsgi
from cheroot.test import webtest
class CherootWebCase(webtest.WebCase):
    """Helper class for a web app test suite."""
    script_name = ''
    scheme = 'http'
    available_servers = {'wsgi': cheroot.wsgi.Server, 'native': cheroot.server.HTTPServer}

    @classmethod
    def setup_class(cls):
        """Create and run one HTTP server per class."""
        conf = config.copy()
        conf.update(getattr(cls, 'config', {}))
        s_class = conf.pop('server', 'wsgi')
        server_factory = cls.available_servers.get(s_class)
        if server_factory is None:
            raise RuntimeError('Unknown server in config: %s' % conf['server'])
        cls.httpserver = server_factory(**conf)
        cls.HOST, cls.PORT = cls.httpserver.bind_addr
        if cls.httpserver.ssl_adapter is None:
            ssl = ''
            cls.scheme = 'http'
        else:
            ssl = ' (ssl)'
            cls.HTTP_CONN = http.client.HTTPSConnection
            cls.scheme = 'https'
        v = sys.version.split()[0]
        log.info('Python version used to run this test script: %s', v)
        log.info('Cheroot version: %s', cheroot.__version__)
        log.info('HTTP server version: %s%s', cls.httpserver.protocol, ssl)
        log.info('PID: %s', os.getpid())
        if hasattr(cls, 'setup_server'):
            cls.setup_server()
            cls.start()

    @classmethod
    def teardown_class(cls):
        """Cleanup HTTP server."""
        if hasattr(cls, 'setup_server'):
            cls.stop()

    @classmethod
    def start(cls):
        """Load and start the HTTP server."""
        threading.Thread(target=cls.httpserver.safe_start).start()
        while not cls.httpserver.ready:
            time.sleep(0.1)

    @classmethod
    def stop(cls):
        """Terminate HTTP server."""
        cls.httpserver.stop()
        td = getattr(cls, 'teardown', None)
        if td:
            td()
    date_tolerance = 2

    def assertEqualDates(self, dt1, dt2, seconds=None):
        """Assert ``abs(dt1 - dt2)`` is within ``Y`` seconds."""
        if seconds is None:
            seconds = self.date_tolerance
        if dt1 > dt2:
            diff = dt1 - dt2
        else:
            diff = dt2 - dt1
        if not diff < datetime.timedelta(seconds=seconds):
            raise AssertionError('%r and %r are not within %r seconds.' % (dt1, dt2, seconds))