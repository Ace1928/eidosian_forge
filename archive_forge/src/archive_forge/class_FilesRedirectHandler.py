from __future__ import annotations
import functools
import inspect
import ipaddress
import json
import mimetypes
import os
import re
import types
import warnings
from http.client import responses
from logging import Logger
from typing import TYPE_CHECKING, Any, Awaitable, Coroutine, Sequence, cast
from urllib.parse import urlparse
import prometheus_client
from jinja2 import TemplateNotFound
from jupyter_core.paths import is_hidden
from jupyter_events import EventLogger
from tornado import web
from tornado.log import app_log
from traitlets.config import Application
import jupyter_server
from jupyter_server import CallContext
from jupyter_server._sysinfo import get_sys_info
from jupyter_server._tz import utcnow
from jupyter_server.auth.decorator import allow_unauthenticated, authorized
from jupyter_server.auth.identity import User
from jupyter_server.i18n import combine_translations
from jupyter_server.services.security import csp_report_uri
from jupyter_server.utils import (
class FilesRedirectHandler(JupyterHandler):
    """Handler for redirecting relative URLs to the /files/ handler"""

    @staticmethod
    async def redirect_to_files(self: Any, path: str) -> None:
        """make redirect logic a reusable static method

        so it can be called from other handlers.
        """
        cm = self.contents_manager
        if await ensure_async(cm.dir_exists(path)):
            url = url_path_join(self.base_url, 'tree', url_escape(path))
        else:
            orig_path = path
            parts = path.split('/')
            if not await ensure_async(cm.file_exists(path=path)) and 'files' in parts:
                self.log.warning('Deprecated files/ URL: %s', orig_path)
                parts.remove('files')
                path = '/'.join(parts)
            if not await ensure_async(cm.file_exists(path=path)):
                raise web.HTTPError(404)
            url = url_path_join(self.base_url, 'files', url_escape(path))
        self.log.debug('Redirecting %s to %s', self.request.path, url)
        self.redirect(url)

    @allow_unauthenticated
    async def get(self, path: str='') -> None:
        return await self.redirect_to_files(self, path)