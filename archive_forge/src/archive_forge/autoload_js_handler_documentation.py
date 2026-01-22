from __future__ import annotations
import logging # isort:skip
from urllib.parse import urlparse
from bokeh.core.templates import AUTOLOAD_JS
from bokeh.embed.bundle import Script, bundle_for_objs_and_resources
from bokeh.embed.elements import script_for_render_items
from bokeh.embed.util import RenderItem
from .session_handler import SessionHandler
Browsers make OPTIONS requests under the hood before a GET request