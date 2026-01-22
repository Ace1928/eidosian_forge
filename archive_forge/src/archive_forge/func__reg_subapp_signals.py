import asyncio
import logging
import warnings
from functools import partial, update_wrapper
from typing import (
from aiosignal import Signal
from frozenlist import FrozenList
from . import hdrs
from .abc import (
from .helpers import DEBUG, AppKey
from .http_parser import RawRequestMessage
from .log import web_logger
from .streams import StreamReader
from .typedefs import Middleware
from .web_exceptions import NotAppKeyWarning
from .web_log import AccessLogger
from .web_middlewares import _fix_request_current_app
from .web_protocol import RequestHandler
from .web_request import Request
from .web_response import StreamResponse
from .web_routedef import AbstractRouteDef
from .web_server import Server
from .web_urldispatcher import (
def _reg_subapp_signals(self, subapp: 'Application') -> None:

    def reg_handler(signame: str) -> None:
        subsig = getattr(subapp, signame)

        async def handler(app: 'Application') -> None:
            await subsig.send(subapp)
        appsig = getattr(self, signame)
        appsig.append(handler)
    reg_handler('on_startup')
    reg_handler('on_shutdown')
    reg_handler('on_cleanup')