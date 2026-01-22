from typing import Optional, Text
from jupyter_core.utils import ensure_async
from jupyter_server.base.handlers import APIHandler, JupyterHandler
from jupyter_server.utils import url_path_join as ujoin
from tornado import web
from tornado.websocket import WebSocketHandler
from .manager import LanguageServerManager
from .schema import SERVERS_RESPONSE
from .specs.utils import censored_spec
class LanguageServersHandler(BaseHandler):
    """Reports the status of all current servers

    Response should conform to schema in schema/servers.schema.json
    """
    auth_resource = AUTH_RESOURCE
    validator = SERVERS_RESPONSE

    @web.authenticated
    @authorized
    async def get(self):
        """finish with the JSON representations of the sessions"""
        await self.manager.ready()
        response = {'version': 2, 'sessions': {language_server: session.to_json() for language_server, session in self.manager.sessions.items()}, 'specs': {key: censored_spec(spec) for key, spec in self.manager.all_language_servers.items()}}
        errors = list(self.validator.iter_errors(response))
        if errors:
            self.log.warning('{} validation errors: {}'.format(len(errors), errors))
        self.finish(response)