import os
import sys
from datetime import datetime
from functools import partial
import time
from gevent.pool import Pool
from gevent.server import StreamServer
from gevent import hub, monkey, socket, pywsgi
import gunicorn
from gunicorn.http.wsgi import base_environ
from gunicorn.sock import ssl_context
from gunicorn.workers.base_async import AsyncWorker
class GeventPyWSGIWorker(GeventWorker):
    """The Gevent StreamServer based workers."""
    server_class = PyWSGIServer
    wsgi_handler = PyWSGIHandler