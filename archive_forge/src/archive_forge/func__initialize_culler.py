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
def _initialize_culler(self) -> None:
    """Start culler if 'cull_inactive_timeout' is greater than zero.
        Regardless of that value, set flag that we've been here.
        """
    if not self._initialized_culler and self.cull_inactive_timeout > 0:
        if self._culler_callback is None:
            _ = IOLoop.current()
            if self.cull_interval <= 0:
                self.log.warning("Invalid value for 'cull_interval' detected (%s) - using default value (%s).", self.cull_interval, self.cull_interval_default)
                self.cull_interval = self.cull_interval_default
            self._culler_callback = PeriodicCallback(self._cull_terminals, 1000 * self.cull_interval)
            self.log.info('Culling terminals with inactivity > %s seconds at %s second intervals ...', self.cull_inactive_timeout, self.cull_interval)
            self._culler_callback.start()
    self._initialized_culler = True