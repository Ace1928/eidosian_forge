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
class CPWebCase(webtest.WebCase):
    script_name = ''
    scheme = 'http'
    available_servers = {'wsgi': LocalWSGISupervisor, 'wsgi_u': get_wsgi_u_supervisor, 'native': NativeServerSupervisor, 'cpmodpy': get_cpmodpy_supervisor, 'modpygw': get_modpygw_supervisor, 'modwsgi': get_modwsgi_supervisor, 'modfcgid': get_modfcgid_supervisor, 'modfastcgi': get_modfastcgi_supervisor}
    default_server = 'wsgi'

    @classmethod
    def _setup_server(cls, supervisor, conf):
        v = sys.version.split()[0]
        log.info('Python version used to run this test script: %s' % v)
        log.info('CherryPy version: %s' % cherrypy.__version__)
        if supervisor.scheme == 'https':
            ssl = ' (ssl)'
        else:
            ssl = ''
        log.info('HTTP server version: %s%s' % (supervisor.protocol, ssl))
        log.info('PID: %s' % os.getpid())
        cherrypy.server.using_apache = supervisor.using_apache
        cherrypy.server.using_wsgi = supervisor.using_wsgi
        if sys.platform[:4] == 'java':
            cherrypy.config.update({'server.nodelay': False})
        if isinstance(conf, text_or_bytes):
            parser = cherrypy.lib.reprconf.Parser()
            conf = parser.dict_from_file(conf).get('global', {})
        else:
            conf = conf or {}
        baseconf = conf.copy()
        baseconf.update({'server.socket_host': supervisor.host, 'server.socket_port': supervisor.port, 'server.protocol_version': supervisor.protocol, 'environment': 'test_suite'})
        if supervisor.scheme == 'https':
            baseconf['server.ssl_certificate'] = serverpem
            baseconf['server.ssl_private_key'] = serverpem
        if supervisor.scheme == 'https':
            webtest.WebCase.HTTP_CONN = HTTPSConnection
        return baseconf

    @classmethod
    def setup_class(cls):
        """"""
        conf = {'scheme': 'http', 'protocol': 'HTTP/1.1', 'port': 54583, 'host': '127.0.0.1', 'validate': False, 'server': 'wsgi'}
        supervisor_factory = cls.available_servers.get(conf.get('server', 'wsgi'))
        if supervisor_factory is None:
            raise RuntimeError('Unknown server in config: %s' % conf['server'])
        supervisor = supervisor_factory(**conf)
        cherrypy.config.reset()
        baseconf = cls._setup_server(supervisor, conf)
        cherrypy.config.update(baseconf)
        setup_client()
        if hasattr(cls, 'setup_server'):
            cherrypy.tree = cherrypy._cptree.Tree()
            cherrypy.server.httpserver = None
            cls.setup_server()
            cherrypy.tree.mount(gctools.GCRoot(), '/gc')
            cls.do_gc_test = True
            supervisor.start(cls.__module__)
        cls.supervisor = supervisor

    @classmethod
    def teardown_class(cls):
        """"""
        if hasattr(cls, 'setup_server'):
            cls.supervisor.stop()
    do_gc_test = False

    def test_gc(self):
        if not self.do_gc_test:
            return
        self.getPage('/gc/stats')
        try:
            self.assertBody('Statistics:')
        except Exception:
            'Failures occur intermittently. See #1420'

    def prefix(self):
        return self.script_name.rstrip('/')

    def base(self):
        if self.scheme == 'http' and self.PORT == 80 or (self.scheme == 'https' and self.PORT == 443):
            port = ''
        else:
            port = ':%s' % self.PORT
        return '%s://%s%s%s' % (self.scheme, self.HOST, port, self.script_name.rstrip('/'))

    def exit(self):
        sys.exit()

    def getPage(self, url, *args, **kwargs):
        """Open the url.
        """
        if self.script_name:
            url = httputil.urljoin(self.script_name, url)
        return webtest.WebCase.getPage(self, url, *args, **kwargs)

    def skip(self, msg='skipped '):
        pytest.skip(msg)

    def assertErrorPage(self, status, message=None, pattern=''):
        """Compare the response body with a built in error page.

        The function will optionally look for the regexp pattern,
        within the exception embedded in the error page."""
        page = cherrypy._cperror.get_error_page(status, message=message)

        def esc(text):
            return re.escape(ntob(text))
        epage = re.escape(page)
        epage = epage.replace(esc('<pre id="traceback"></pre>'), esc('<pre id="traceback">') + b'(.*)' + esc('</pre>'))
        m = re.match(epage, self.body, re.DOTALL)
        if not m:
            self._handlewebError('Error page does not match; expected:\n' + page)
            return
        if pattern is None:
            if m and m.group(1):
                self._handlewebError('Error page contains traceback')
        elif m is None or not re.search(ntob(re.escape(pattern), self.encoding), m.group(1)):
            msg = 'Error page does not contain %s in traceback'
            self._handlewebError(msg % repr(pattern))
    date_tolerance = 2

    def assertEqualDates(self, dt1, dt2, seconds=None):
        """Assert abs(dt1 - dt2) is within Y seconds."""
        if seconds is None:
            seconds = self.date_tolerance
        if dt1 > dt2:
            diff = dt1 - dt2
        else:
            diff = dt2 - dt1
        if not diff < datetime.timedelta(seconds=seconds):
            raise AssertionError('%r and %r are not within %r seconds.' % (dt1, dt2, seconds))