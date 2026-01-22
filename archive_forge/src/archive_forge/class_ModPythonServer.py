import cherrypy
import io
import logging
import os
import re
import sys
from more_itertools import always_iterable
import cherrypy
from cherrypy._cperror import format_exc, bare_error
from cherrypy.lib import httputil
class ModPythonServer(object):
    template = '\n# Apache2 server configuration file for running CherryPy with mod_python.\n\nDocumentRoot "/"\nListen %(port)s\nLoadModule python_module modules/mod_python.so\n\n<Location %(loc)s>\n    SetHandler python-program\n    PythonHandler %(handler)s\n    PythonDebug On\n%(opts)s\n</Location>\n'

    def __init__(self, loc='/', port=80, opts=None, apache_path='apache', handler='cherrypy._cpmodpy::handler'):
        self.loc = loc
        self.port = port
        self.opts = opts
        self.apache_path = apache_path
        self.handler = handler

    def start(self):
        opts = ''.join(['    PythonOption %s %s\n' % (k, v) for k, v in self.opts])
        conf_data = self.template % {'port': self.port, 'loc': self.loc, 'opts': opts, 'handler': self.handler}
        mpconf = os.path.join(os.path.dirname(__file__), 'cpmodpy.conf')
        with open(mpconf, 'wb') as f:
            f.write(conf_data)
        response = read_process(self.apache_path, '-k start -f %s' % mpconf)
        self.ready = True
        return response

    def stop(self):
        os.popen('apache -k stop')
        self.ready = False