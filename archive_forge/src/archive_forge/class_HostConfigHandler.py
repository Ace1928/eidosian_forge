from __future__ import annotations
import os
from typing import Final
import tornado.web
from streamlit import config, file_util
from streamlit.logger import get_logger
from streamlit.runtime.runtime_util import serialize_forward_msg
from streamlit.web.server.server_util import emit_endpoint_deprecation_notice
class HostConfigHandler(_SpecialRequestHandler):

    def initialize(self):
        self._allowed_origins = _DEFAULT_ALLOWED_MESSAGE_ORIGINS.copy()
        if config.get_option('global.developmentMode') and 'http://localhost' not in self._allowed_origins:
            self._allowed_origins.append('http://localhost')

    async def get(self) -> None:
        self.write({'allowedOrigins': self._allowed_origins, 'useExternalAuthToken': False, 'enableCustomParentMessages': False})
        self.set_status(200)