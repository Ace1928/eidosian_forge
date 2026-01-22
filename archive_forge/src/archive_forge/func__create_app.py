from __future__ import annotations
import errno
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Final
import tornado.concurrent
import tornado.locks
import tornado.netutil
import tornado.web
import tornado.websocket
from tornado.httpserver import HTTPServer
from streamlit import cli_util, config, file_util, source_util, util
from streamlit.components.v1.components import ComponentRegistry
from streamlit.config_option import ConfigOption
from streamlit.logger import get_logger
from streamlit.runtime import Runtime, RuntimeConfig, RuntimeState
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
from streamlit.runtime.runtime_util import get_max_message_size_bytes
from streamlit.web.cache_storage_manager_config import (
from streamlit.web.server.app_static_file_handler import AppStaticFileHandler
from streamlit.web.server.browser_websocket_handler import BrowserWebSocketHandler
from streamlit.web.server.component_request_handler import ComponentRequestHandler
from streamlit.web.server.media_file_handler import MediaFileHandler
from streamlit.web.server.routes import (
from streamlit.web.server.server_util import DEVELOPMENT_PORT, make_url_path_regex
from streamlit.web.server.stats_request_handler import StatsRequestHandler
from streamlit.web.server.upload_file_request_handler import UploadFileRequestHandler
def _create_app(self) -> tornado.web.Application:
    """Create our tornado web app."""
    base = config.get_option('server.baseUrlPath')
    routes: list[Any] = [(make_url_path_regex(base, STREAM_ENDPOINT), BrowserWebSocketHandler, dict(runtime=self._runtime)), (make_url_path_regex(base, HEALTH_ENDPOINT), HealthHandler, dict(callback=lambda: self._runtime.is_ready_for_browser_connection)), (make_url_path_regex(base, MESSAGE_ENDPOINT), MessageCacheHandler, dict(cache=self._runtime.message_cache)), (make_url_path_regex(base, METRIC_ENDPOINT), StatsRequestHandler, dict(stats_manager=self._runtime.stats_mgr)), (make_url_path_regex(base, HOST_CONFIG_ENDPOINT), HostConfigHandler), (make_url_path_regex(base, f'{UPLOAD_FILE_ENDPOINT}/(?P<session_id>[^/]+)/(?P<file_id>[^/]+)'), UploadFileRequestHandler, dict(file_mgr=self._runtime.uploaded_file_mgr, is_active_session=self._runtime.is_active_session)), (make_url_path_regex(base, f'{MEDIA_ENDPOINT}/(.*)'), MediaFileHandler, {'path': ''}), (make_url_path_regex(base, 'component/(.*)'), ComponentRequestHandler, dict(registry=ComponentRegistry.instance()))]
    if config.get_option('server.scriptHealthCheckEnabled'):
        routes.extend([(make_url_path_regex(base, SCRIPT_HEALTH_CHECK_ENDPOINT), HealthHandler, dict(callback=lambda: self._runtime.does_script_run_without_error()))])
    if config.get_option('server.enableStaticServing'):
        routes.extend([(make_url_path_regex(base, 'app/static/(.*)'), AppStaticFileHandler, {'path': file_util.get_app_static_dir(self.main_script_path)})])
    if config.get_option('global.developmentMode'):
        _LOGGER.debug('Serving static content from the Node dev server')
    else:
        static_path = file_util.get_static_dir()
        _LOGGER.debug('Serving static content from %s', static_path)
        routes.extend([(make_url_path_regex(base, '(.*)'), StaticFileHandler, {'path': '%s/' % static_path, 'default_filename': 'index.html', 'get_pages': lambda: {page_info['page_name'] for page_info in source_util.get_pages(self.main_script_path).values()}}), (make_url_path_regex(base, trailing_slash=False), AddSlashHandler)])
    return tornado.web.Application(routes, cookie_secret=config.get_option('server.cookieSecret'), xsrf_cookies=config.get_option('server.enableXsrfProtection'), websocket_max_message_size=get_max_message_size_bytes(), **TORNADO_SETTINGS)