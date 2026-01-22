from typing import Optional, Text
from jupyter_core.utils import ensure_async
from jupyter_server.base.handlers import APIHandler, JupyterHandler
from jupyter_server.utils import url_path_join as ujoin
from tornado import web
from tornado.websocket import WebSocketHandler
from .manager import LanguageServerManager
from .schema import SERVERS_RESPONSE
from .specs.utils import censored_spec
class LanguageServerWebSocketHandler(WebSocketMixin, WebSocketHandler, BaseJupyterHandler):
    """Setup tornado websocket to route to language server sessions.

    The logic of `get` and `pre_get` methods is derived from jupyter-server ws handlers,
    and should be kept in sync to follow best practice established by upstream; see:
    https://github.com/jupyter-server/jupyter_server/blob/v2.12.5/jupyter_server/services/kernels/websocket.py#L36
    """
    auth_resource = AUTH_RESOURCE
    language_server: Optional[Text] = None

    async def pre_get(self):
        """Handle a pre_get."""
        user = self.current_user
        if user is None:
            self.log.warning("Couldn't authenticate WebSocket connection")
            raise web.HTTPError(403)
        if not hasattr(self, 'authorizer'):
            return
        is_authorized = await ensure_async(self.authorizer.is_authorized(self, user, 'execute', AUTH_RESOURCE))
        if not is_authorized:
            raise web.HTTPError(403)

    async def get(self, *args, **kwargs):
        """Get an event socket."""
        await self.pre_get()
        res = super().get(*args, **kwargs)
        if res is not None:
            await res

    async def open(self, language_server):
        await self.manager.ready()
        self.language_server = language_server
        self.manager.subscribe(self)
        self.log.debug('[{}] Opened a handler'.format(self.language_server))
        super().open()

    async def on_message(self, message):
        self.log.debug('[{}] Handling a message'.format(self.language_server))
        await self.manager.on_client_message(message, self)

    def on_close(self):
        self.manager.unsubscribe(self)
        self.log.debug('[{}] Closed a handler'.format(self.language_server))