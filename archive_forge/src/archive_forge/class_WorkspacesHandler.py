from __future__ import annotations
import hashlib
import json
import re
import unicodedata
import urllib
from pathlib import Path
from typing import Any
from jupyter_server import _tz as tz
from jupyter_server.base.handlers import APIHandler
from jupyter_server.extension.handler import ExtensionHandlerJinjaMixin, ExtensionHandlerMixin
from jupyter_server.utils import url_path_join as ujoin
from tornado import web
from traitlets.config import LoggingConfigurable
class WorkspacesHandler(ExtensionHandlerMixin, ExtensionHandlerJinjaMixin, APIHandler):
    """A workspaces API handler."""

    def initialize(self, name: str, manager: WorkspacesManager, **kwargs: Any) -> None:
        """Initialize the handler."""
        super().initialize(name)
        self.manager = manager

    @web.authenticated
    def delete(self, space_name: str) -> None:
        """Remove a workspace"""
        if not space_name:
            raise web.HTTPError(400, 'Workspace name is required for DELETE')
        try:
            self.manager.delete(space_name)
            return self.set_status(204)
        except FileNotFoundError as e:
            raise web.HTTPError(404, str(e)) from e
        except Exception as e:
            raise web.HTTPError(500, str(e)) from e

    @web.authenticated
    async def get(self, space_name: str='') -> Any:
        """Get workspace(s) data"""
        try:
            if not space_name:
                workspaces = self.manager.list_workspaces()
                ids = []
                values = []
                for workspace in workspaces:
                    ids.append(workspace['metadata']['id'])
                    values.append(workspace)
                return self.finish(json.dumps({'workspaces': {'ids': ids, 'values': values}}))
            workspace = self.manager.load(space_name)
            return self.finish(json.dumps(workspace))
        except Exception as e:
            raise web.HTTPError(500, str(e)) from e

    @web.authenticated
    def put(self, space_name: str='') -> None:
        """Update workspace data"""
        if not space_name:
            raise web.HTTPError(400, 'Workspace name is required for PUT.')
        raw = self.request.body.strip().decode('utf-8')
        try:
            self.manager.save(space_name, raw)
        except ValueError as e:
            raise web.HTTPError(400, str(e)) from e
        except Exception as e:
            raise web.HTTPError(500, str(e)) from e
        self.set_status(204)