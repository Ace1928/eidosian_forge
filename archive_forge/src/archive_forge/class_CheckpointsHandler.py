import json
from http import HTTPStatus
from typing import Any, Dict, List
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server.auth.decorator import allow_unauthenticated, authorized
from jupyter_server.base.handlers import APIHandler, JupyterHandler, path_regex
from jupyter_server.utils import url_escape, url_path_join
class CheckpointsHandler(ContentsAPIHandler):
    """A checkpoints API handler."""

    @web.authenticated
    @authorized
    async def get(self, path=''):
        """get lists checkpoints for a file"""
        cm = self.contents_manager
        checkpoints = await ensure_async(cm.list_checkpoints(path))
        data = json.dumps(checkpoints, default=json_default)
        self.finish(data)

    @web.authenticated
    @authorized
    async def post(self, path=''):
        """post creates a new checkpoint"""
        cm = self.contents_manager
        checkpoint = await ensure_async(cm.create_checkpoint(path))
        data = json.dumps(checkpoint, default=json_default)
        location = url_path_join(self.base_url, 'api/contents', url_escape(path), 'checkpoints', url_escape(checkpoint['id']))
        self.set_header('Location', location)
        self.set_status(201)
        self.finish(data)