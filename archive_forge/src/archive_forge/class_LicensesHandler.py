from __future__ import annotations
import asyncio
import csv
import io
import json
import mimetypes
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any
from jupyter_server.base.handlers import APIHandler
from tornado import web
from traitlets import List, Unicode
from traitlets.config import LoggingConfigurable
from .config import get_federated_extensions
class LicensesHandler(APIHandler):
    """A handler for serving licenses used by the application"""

    def initialize(self, manager: LicensesManager) -> None:
        """Initialize the handler."""
        super().initialize()
        self.manager = manager

    @web.authenticated
    async def get(self, _args: Any) -> None:
        """Return all the frontend licenses"""
        full_text = bool(json.loads(self.get_argument('full_text', 'true')))
        report_format = self.get_argument('format', 'json')
        bundles_pattern = self.get_argument('bundles', '.*')
        download = bool(json.loads(self.get_argument('download', '0')))
        report, mime = await self.manager.report_async(report_format=report_format, bundles_pattern=bundles_pattern, full_text=full_text)
        if TYPE_CHECKING:
            from .app import LabServerApp
            assert isinstance(self.manager.parent, LabServerApp)
        if download:
            filename = '{}-licenses{}'.format(self.manager.parent.app_name.lower(), mimetypes.guess_extension(mime))
            self.set_attachment_header(filename)
        self.write(report)
        await self.finish(_mime_type=mime)

    async def finish(self, _mime_type: str, *args: Any, **kwargs: Any) -> Any:
        """Overload the regular finish, which (sensibly) always sets JSON"""
        self.update_api_activity()
        self.set_header('Content-Type', _mime_type)
        return await super(APIHandler, self).finish(*args, **kwargs)