import json
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server.auth.decorator import authorized
from jupyter_server.utils import url_escape, url_path_join
from ...base.handlers import APIHandler
from .websocket import KernelWebsocketHandler
class MainKernelHandler(KernelsAPIHandler):
    """The root kernel handler."""

    @web.authenticated
    @authorized
    async def get(self):
        """Get the list of running kernels."""
        km = self.kernel_manager
        kernels = await ensure_async(km.list_kernels())
        self.finish(json.dumps(kernels, default=json_default))

    @web.authenticated
    @authorized
    async def post(self):
        """Start a kernel."""
        km = self.kernel_manager
        model = self.get_json_body()
        if model is None:
            model = {'name': km.default_kernel_name}
        else:
            model.setdefault('name', km.default_kernel_name)
        kernel_id = await ensure_async(km.start_kernel(kernel_name=model['name'], path=model.get('path')))
        model = await ensure_async(km.kernel_model(kernel_id))
        location = url_path_join(self.base_url, 'api', 'kernels', url_escape(kernel_id))
        self.set_header('Location', location)
        self.set_status(201)
        self.finish(json.dumps(model, default=json_default))