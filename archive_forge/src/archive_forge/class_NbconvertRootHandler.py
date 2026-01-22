import asyncio
import json
from anyio.to_thread import run_sync
from tornado import web
from jupyter_server.auth.decorator import authorized
from ...base.handlers import APIHandler
class NbconvertRootHandler(APIHandler):
    """The nbconvert root API handler."""
    auth_resource = AUTH_RESOURCE
    _exporter_lock: asyncio.Lock

    def initialize(self, **kwargs):
        """Initialize an nbconvert root handler."""
        super().initialize(**kwargs)
        if not hasattr(self.__class__, '_exporter_lock'):
            self.__class__._exporter_lock = asyncio.Lock()
        self._exporter_lock = self.__class__._exporter_lock

    @web.authenticated
    @authorized
    async def get(self):
        """Get the list of nbconvert exporters."""
        try:
            from nbconvert.exporters import base
        except ImportError as e:
            raise web.HTTPError(500, 'Could not import nbconvert: %s' % e) from e
        res = {}
        exporters = await run_sync(base.get_export_names)
        async with self._exporter_lock:
            for exporter_name in exporters:
                try:
                    exporter_class = await run_sync(base.get_exporter, exporter_name)
                except ValueError:
                    continue
                res[exporter_name] = {'output_mimetype': exporter_class.output_mimetype}
        self.finish(json.dumps(res))