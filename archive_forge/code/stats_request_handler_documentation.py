from __future__ import annotations
from typing import TYPE_CHECKING
import tornado.web
from streamlit.runtime.stats import CacheStat, StatsManager
from streamlit.web.server.server_util import emit_endpoint_deprecation_notice
/OPTIONS handler for preflight CORS checks.