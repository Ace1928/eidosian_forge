from typing import Optional, Text
from jupyter_core.utils import ensure_async
from jupyter_server.base.handlers import APIHandler, JupyterHandler
from jupyter_server.utils import url_path_join as ujoin
from tornado import web
from tornado.websocket import WebSocketHandler
from .manager import LanguageServerManager
from .schema import SERVERS_RESPONSE
from .specs.utils import censored_spec
class BaseJupyterHandler(JupyterHandler):
    manager = None

    def initialize(self, manager: LanguageServerManager):
        self.manager = manager