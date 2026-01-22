from __future__ import annotations
import os
import pathlib
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from jupyter_server.base.handlers import FileFindHandler, JupyterHandler
from jupyter_server.extension.handler import ExtensionHandlerJinjaMixin, ExtensionHandlerMixin
from jupyter_server.utils import url_path_join as ujoin
from tornado import template, web
from .config import LabConfig, get_page_config, recursive_update
from .licenses_handler import LicensesHandler, LicensesManager
from .listings_handler import ListingsHandler, fetch_listings
from .settings_handler import SettingsHandler
from .themes_handler import ThemesHandler
from .translations_handler import TranslationsHandler
from .workspaces_handler import WorkspacesHandler, WorkspacesManager
class NotFoundHandler(LabHandler):
    """A handler for page not found."""

    @lru_cache
    def get_page_config(self) -> dict[str, Any]:
        """Get the page config."""
        page_config = super().get_page_config().copy()
        page_config['notFoundUrl'] = self.request.path
        return page_config