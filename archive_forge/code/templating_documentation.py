from __future__ import annotations
import typing
import warnings
from os import PathLike
from starlette.background import BackgroundTask
from starlette.datastructures import URL
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.types import Receive, Scope, Send

    templates = Jinja2Templates("templates")

    return templates.TemplateResponse("index.html", {"request": request})
    