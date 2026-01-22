from __future__ import annotations
import typing as t
from datetime import timedelta
from jupyter_server._tz import isoformat, utcnow
from jupyter_server.prometheus import metrics
from terminado.management import NamedTermManager, PtyWithClients
from tornado import web
from tornado.ioloop import IOLoop, PeriodicCallback
from traitlets import Integer
from traitlets.config import LoggingConfigurable
def _check_terminal(self, name: str) -> None:
    """Check a that terminal 'name' exists and raise 404 if not."""
    if name not in self.terminals:
        raise web.HTTPError(404, 'Terminal not found: %s' % name)