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
class CPProcess(object):
    pid_file = os.path.join(thisdir, 'test.pid')
    config_file = os.path.join(thisdir, 'test.conf')
    config_template = "[global]\nserver.socket_host: '%(host)s'\nserver.socket_port: %(port)s\nchecker.on: False\nlog.screen: False\nlog.error_file: r'%(error_log)s'\nlog.access_file: r'%(access_log)s'\n%(ssl)s\n%(extra)s\n"
    error_log = os.path.join(thisdir, 'test.error.log')
    access_log = os.path.join(thisdir, 'test.access.log')

    def __init__(self, wait=False, daemonize=False, ssl=False, socket_host=None, socket_port=None):
        self.wait = wait
        self.daemonize = daemonize
        self.ssl = ssl
        self.host = socket_host or cherrypy.server.socket_host
        self.port = socket_port or cherrypy.server.socket_port

    def write_conf(self, extra=''):
        if self.ssl:
            serverpem = os.path.join(thisdir, 'test.pem')
            ssl = "\nserver.ssl_certificate: r'%s'\nserver.ssl_private_key: r'%s'\n" % (serverpem, serverpem)
        else:
            ssl = ''
        conf = self.config_template % {'host': self.host, 'port': self.port, 'error_log': self.error_log, 'access_log': self.access_log, 'ssl': ssl, 'extra': extra}
        with io.open(self.config_file, 'w', encoding='utf-8') as f:
            f.write(str(conf))

    def start(self, imports=None):
        """Start cherryd in a subprocess."""
        portend.free(self.host, self.port, timeout=1)
        args = ['-m', 'cherrypy', '-c', self.config_file, '-p', self.pid_file]
        '\n        Command for running cherryd server with autoreload enabled\n\n        Using\n\n        ```\n        [\'-c\',\n         "__requires__ = \'CherryPy\'; \\\n         import pkg_resources, re, sys; \\\n         sys.argv[0] = re.sub(r\'(-script\\.pyw?|\\.exe)?$\', \'\', sys.argv[0]); \\\n         sys.exit(\\\n            pkg_resources.load_entry_point(\\\n                \'CherryPy\', \'console_scripts\', \'cherryd\')())"]\n        ```\n\n        doesn\'t work as it\'s impossible to reconstruct the `-c`\'s contents.\n        Ref: https://github.com/cherrypy/cherrypy/issues/1545\n        '
        if not isinstance(imports, (list, tuple)):
            imports = [imports]
        for i in imports:
            if i:
                args.append('-i')
                args.append(i)
        if self.daemonize:
            args.append('-d')
        env = os.environ.copy()
        grandparentdir = os.path.abspath(os.path.join(thisdir, '..', '..'))
        if env.get('PYTHONPATH', ''):
            env['PYTHONPATH'] = os.pathsep.join((grandparentdir, env['PYTHONPATH']))
        else:
            env['PYTHONPATH'] = grandparentdir
        self._proc = subprocess.Popen([sys.executable] + args, env=env)
        if self.wait:
            self.exit_code = self._proc.wait()
        else:
            portend.occupied(self.host, self.port, timeout=5)
        if self.daemonize:
            time.sleep(2)
        else:
            time.sleep(1)

    def get_pid(self):
        if self.daemonize:
            with open(self.pid_file, 'rb') as f:
                return int(f.read())
        return self._proc.pid

    def join(self):
        """Wait for the process to exit."""
        if self.daemonize:
            return self._join_daemon()
        self._proc.wait()

    def _join_daemon(self):
        with contextlib.suppress(IOError):
            os.waitpid(self.get_pid(), 0)