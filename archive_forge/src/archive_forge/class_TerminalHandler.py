from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from jupyter_server.auth.decorator import authorized
from jupyter_server.base.handlers import APIHandler
from tornado import web
from .base import TerminalsMixin
class TerminalHandler(TerminalsMixin, TerminalAPIHandler):
    """A handler for a specific terminal."""
    SUPPORTED_METHODS = ('GET', 'DELETE', 'OPTIONS')

    @web.authenticated
    @authorized
    def get(self, name: str) -> None:
        """Get a terminal by name."""
        model = self.terminal_manager.get(name)
        self.finish(json.dumps(model))

    @web.authenticated
    @authorized
    async def delete(self, name: str) -> None:
        """Remove a terminal by name."""
        await self.terminal_manager.terminate(name, force=True)
        self.set_status(204)
        self.finish()