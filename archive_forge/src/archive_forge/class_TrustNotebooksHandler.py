import json
from http import HTTPStatus
from typing import Any, Dict, List
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server.auth.decorator import allow_unauthenticated, authorized
from jupyter_server.base.handlers import APIHandler, JupyterHandler, path_regex
from jupyter_server.utils import url_escape, url_path_join
class TrustNotebooksHandler(JupyterHandler):
    """Handles trust/signing of notebooks"""

    @web.authenticated
    @authorized(resource=AUTH_RESOURCE)
    async def post(self, path=''):
        """Trust a notebook by path."""
        cm = self.contents_manager
        await ensure_async(cm.trust_notebook(path))
        self.set_status(201)
        self.finish()