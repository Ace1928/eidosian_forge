from __future__ import annotations
import difflib
import typing as t
from ..exceptions import BadRequest
from ..exceptions import HTTPException
from ..utils import cached_property
from ..utils import redirect
class RequestRedirect(HTTPException, RoutingException):
    """Raise if the map requests a redirect. This is for example the case if
    `strict_slashes` are activated and an url that requires a trailing slash.

    The attribute `new_url` contains the absolute destination url.
    """
    code = 308

    def __init__(self, new_url: str) -> None:
        super().__init__(new_url)
        self.new_url = new_url

    def get_response(self, environ: WSGIEnvironment | Request | None=None, scope: dict | None=None) -> Response:
        return redirect(self.new_url, self.code)