from __future__ import annotations
import mimetypes
import os
from pathlib import Path
from typing import Final
import tornado.web
from streamlit.logger import get_logger
class AppStaticFileHandler(tornado.web.StaticFileHandler):

    def initialize(self, path: str, default_filename: str | None=None) -> None:
        super().initialize(path, default_filename)
        mimetypes.add_type('image/webp', '.webp')

    def validate_absolute_path(self, root: str, absolute_path: str) -> str | None:
        full_path = os.path.realpath(absolute_path)
        if os.path.isdir(full_path):
            raise tornado.web.HTTPError(404)
        if os.path.commonpath([full_path, root]) != root:
            _LOGGER.warning('Serving files outside of the static directory is not supported')
            raise tornado.web.HTTPError(404)
        if os.path.exists(full_path) and os.path.getsize(full_path) > MAX_APP_STATIC_FILE_SIZE:
            raise tornado.web.HTTPError(404, f'File is too large, its size should not exceed {MAX_APP_STATIC_FILE_SIZE} bytes', reason='File is too large')
        return super().validate_absolute_path(root, absolute_path)

    def set_default_headers(self):
        self.set_header('Access-Control-Allow-Origin', '*')

    def set_extra_headers(self, path: str) -> None:
        if Path(path).suffix not in SAFE_APP_STATIC_FILE_EXTENSIONS:
            self.set_header('Content-Type', 'text/plain')
        self.set_header('X-Content-Type-Options', 'nosniff')