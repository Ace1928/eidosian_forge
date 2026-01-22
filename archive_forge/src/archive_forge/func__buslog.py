from threading import local as _local
from ._cperror import (
from . import _cpdispatch as dispatch
from ._cptools import default_toolbox as tools, Tool
from ._helper import expose, popargs, url
from . import _cprequest, _cpserver, _cptree, _cplogging, _cpconfig
import cherrypy.lib.httputil as _httputil
from ._cptree import Application
from . import _cpwsgi as wsgi
from . import process
from . import _cpchecker
@engine.subscribe('log')
def _buslog(msg, level):
    log.error(msg, 'ENGINE', severity=level)