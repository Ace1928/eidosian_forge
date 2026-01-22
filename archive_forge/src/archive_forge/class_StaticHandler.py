from __future__ import annotations
import logging # isort:skip
from tornado.web import StaticFileHandler
from bokeh.settings import settings
class StaticHandler(StaticFileHandler):
    """ Implements a custom Tornado static file handler for BokehJS
    JavaScript and CSS resources.

    """

    def __init__(self, tornado_app, *args, **kw) -> None:
        kw['path'] = settings.bokehjs_path()
        super().__init__(tornado_app, *args, **kw)

    @classmethod
    def append_version(cls, path: str) -> str:
        if settings.dev:
            return path
        else:
            version = StaticFileHandler.get_version(dict(static_path=settings.bokehjs_path()), path)
            return f'{path}?v={version}'