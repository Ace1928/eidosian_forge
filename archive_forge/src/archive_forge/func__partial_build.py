from __future__ import annotations
import typing as t
import warnings
from pprint import pformat
from threading import Lock
from urllib.parse import quote
from urllib.parse import urljoin
from urllib.parse import urlunsplit
from .._internal import _get_environ
from .._internal import _wsgi_decoding_dance
from ..datastructures import ImmutableDict
from ..datastructures import MultiDict
from ..exceptions import BadHost
from ..exceptions import HTTPException
from ..exceptions import MethodNotAllowed
from ..exceptions import NotFound
from ..urls import _urlencode
from ..wsgi import get_host
from .converters import DEFAULT_CONVERTERS
from .exceptions import BuildError
from .exceptions import NoMatch
from .exceptions import RequestAliasRedirect
from .exceptions import RequestPath
from .exceptions import RequestRedirect
from .exceptions import WebsocketMismatch
from .matcher import StateMachineMatcher
from .rules import _simple_rule_re
from .rules import Rule
def _partial_build(self, endpoint: str, values: t.Mapping[str, t.Any], method: str | None, append_unknown: bool) -> tuple[str, str, bool] | None:
    """Helper for :meth:`build`.  Returns subdomain and path for the
        rule that accepts this endpoint, values and method.

        :internal:
        """
    if method is None:
        rv = self._partial_build(endpoint, values, self.default_method, append_unknown)
        if rv is not None:
            return rv
    first_match = None
    for rule in self.map._rules_by_endpoint.get(endpoint, ()):
        if rule.suitable_for(values, method):
            build_rv = rule.build(values, append_unknown)
            if build_rv is not None:
                rv = (build_rv[0], build_rv[1], rule.websocket)
                if self.map.host_matching:
                    if rv[0] == self.server_name:
                        return rv
                    elif first_match is None:
                        first_match = rv
                else:
                    return rv
    return first_match