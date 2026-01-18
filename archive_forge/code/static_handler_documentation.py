from __future__ import annotations
import logging # isort:skip
from tornado.web import StaticFileHandler
from bokeh.settings import settings
 Implements a custom Tornado static file handler for BokehJS
    JavaScript and CSS resources.

    