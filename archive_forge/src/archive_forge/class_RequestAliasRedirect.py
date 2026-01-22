from __future__ import annotations
import difflib
import typing as t
from ..exceptions import BadRequest
from ..exceptions import HTTPException
from ..utils import cached_property
from ..utils import redirect
class RequestAliasRedirect(RoutingException):
    """This rule is an alias and wants to redirect to the canonical URL."""

    def __init__(self, matched_values: t.Mapping[str, t.Any], endpoint: str) -> None:
        super().__init__()
        self.matched_values = matched_values
        self.endpoint = endpoint