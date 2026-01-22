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
class Map:
    """The map class stores all the URL rules and some configuration
    parameters.  Some of the configuration values are only stored on the
    `Map` instance since those affect all rules, others are just defaults
    and can be overridden for each rule.  Note that you have to specify all
    arguments besides the `rules` as keyword arguments!

    :param rules: sequence of url rules for this map.
    :param default_subdomain: The default subdomain for rules without a
                              subdomain defined.
    :param strict_slashes: If a rule ends with a slash but the matched
        URL does not, redirect to the URL with a trailing slash.
    :param merge_slashes: Merge consecutive slashes when matching or
        building URLs. Matches will redirect to the normalized URL.
        Slashes in variable parts are not merged.
    :param redirect_defaults: This will redirect to the default rule if it
                              wasn't visited that way. This helps creating
                              unique URLs.
    :param converters: A dict of converters that adds additional converters
                       to the list of converters. If you redefine one
                       converter this will override the original one.
    :param sort_parameters: If set to `True` the url parameters are sorted.
                            See `url_encode` for more details.
    :param sort_key: The sort key function for `url_encode`.
    :param host_matching: if set to `True` it enables the host matching
                          feature and disables the subdomain one.  If
                          enabled the `host` parameter to rules is used
                          instead of the `subdomain` one.

    .. versionchanged:: 3.0
        The ``charset`` and ``encoding_errors`` parameters were removed.

    .. versionchanged:: 1.0
        If ``url_scheme`` is ``ws`` or ``wss``, only WebSocket rules will match.

    .. versionchanged:: 1.0
        The ``merge_slashes`` parameter was added.

    .. versionchanged:: 0.7
        The ``encoding_errors`` and ``host_matching`` parameters were added.

    .. versionchanged:: 0.5
        The ``sort_parameters`` and ``sort_key``  paramters were added.
    """
    default_converters = ImmutableDict(DEFAULT_CONVERTERS)
    lock_class = Lock

    def __init__(self, rules: t.Iterable[RuleFactory] | None=None, default_subdomain: str='', strict_slashes: bool=True, merge_slashes: bool=True, redirect_defaults: bool=True, converters: t.Mapping[str, type[BaseConverter]] | None=None, sort_parameters: bool=False, sort_key: t.Callable[[t.Any], t.Any] | None=None, host_matching: bool=False) -> None:
        self._matcher = StateMachineMatcher(merge_slashes)
        self._rules_by_endpoint: dict[str, list[Rule]] = {}
        self._remap = True
        self._remap_lock = self.lock_class()
        self.default_subdomain = default_subdomain
        self.strict_slashes = strict_slashes
        self.merge_slashes = merge_slashes
        self.redirect_defaults = redirect_defaults
        self.host_matching = host_matching
        self.converters = self.default_converters.copy()
        if converters:
            self.converters.update(converters)
        self.sort_parameters = sort_parameters
        self.sort_key = sort_key
        for rulefactory in rules or ():
            self.add(rulefactory)

    def is_endpoint_expecting(self, endpoint: str, *arguments: str) -> bool:
        """Iterate over all rules and check if the endpoint expects
        the arguments provided.  This is for example useful if you have
        some URLs that expect a language code and others that do not and
        you want to wrap the builder a bit so that the current language
        code is automatically added if not provided but endpoints expect
        it.

        :param endpoint: the endpoint to check.
        :param arguments: this function accepts one or more arguments
                          as positional arguments.  Each one of them is
                          checked.
        """
        self.update()
        arguments = set(arguments)
        for rule in self._rules_by_endpoint[endpoint]:
            if arguments.issubset(rule.arguments):
                return True
        return False

    @property
    def _rules(self) -> list[Rule]:
        return [rule for rules in self._rules_by_endpoint.values() for rule in rules]

    def iter_rules(self, endpoint: str | None=None) -> t.Iterator[Rule]:
        """Iterate over all rules or the rules of an endpoint.

        :param endpoint: if provided only the rules for that endpoint
                         are returned.
        :return: an iterator
        """
        self.update()
        if endpoint is not None:
            return iter(self._rules_by_endpoint[endpoint])
        return iter(self._rules)

    def add(self, rulefactory: RuleFactory) -> None:
        """Add a new rule or factory to the map and bind it.  Requires that the
        rule is not bound to another map.

        :param rulefactory: a :class:`Rule` or :class:`RuleFactory`
        """
        for rule in rulefactory.get_rules(self):
            rule.bind(self)
            if not rule.build_only:
                self._matcher.add(rule)
            self._rules_by_endpoint.setdefault(rule.endpoint, []).append(rule)
        self._remap = True

    def bind(self, server_name: str, script_name: str | None=None, subdomain: str | None=None, url_scheme: str='http', default_method: str='GET', path_info: str | None=None, query_args: t.Mapping[str, t.Any] | str | None=None) -> MapAdapter:
        """Return a new :class:`MapAdapter` with the details specified to the
        call.  Note that `script_name` will default to ``'/'`` if not further
        specified or `None`.  The `server_name` at least is a requirement
        because the HTTP RFC requires absolute URLs for redirects and so all
        redirect exceptions raised by Werkzeug will contain the full canonical
        URL.

        If no path_info is passed to :meth:`match` it will use the default path
        info passed to bind.  While this doesn't really make sense for
        manual bind calls, it's useful if you bind a map to a WSGI
        environment which already contains the path info.

        `subdomain` will default to the `default_subdomain` for this map if
        no defined. If there is no `default_subdomain` you cannot use the
        subdomain feature.

        .. versionchanged:: 1.0
            If ``url_scheme`` is ``ws`` or ``wss``, only WebSocket rules
            will match.

        .. versionchanged:: 0.15
            ``path_info`` defaults to ``'/'`` if ``None``.

        .. versionchanged:: 0.8
            ``query_args`` can be a string.

        .. versionchanged:: 0.7
            Added ``query_args``.
        """
        server_name = server_name.lower()
        if self.host_matching:
            if subdomain is not None:
                raise RuntimeError('host matching enabled and a subdomain was provided')
        elif subdomain is None:
            subdomain = self.default_subdomain
        if script_name is None:
            script_name = '/'
        if path_info is None:
            path_info = '/'
        server_name, port_sep, port = server_name.partition(':')
        try:
            server_name = server_name.encode('idna').decode('ascii')
        except UnicodeError as e:
            raise BadHost() from e
        return MapAdapter(self, f'{server_name}{port_sep}{port}', script_name, subdomain, url_scheme, path_info, default_method, query_args)

    def bind_to_environ(self, environ: WSGIEnvironment | Request, server_name: str | None=None, subdomain: str | None=None) -> MapAdapter:
        """Like :meth:`bind` but you can pass it an WSGI environment and it
        will fetch the information from that dictionary.  Note that because of
        limitations in the protocol there is no way to get the current
        subdomain and real `server_name` from the environment.  If you don't
        provide it, Werkzeug will use `SERVER_NAME` and `SERVER_PORT` (or
        `HTTP_HOST` if provided) as used `server_name` with disabled subdomain
        feature.

        If `subdomain` is `None` but an environment and a server name is
        provided it will calculate the current subdomain automatically.
        Example: `server_name` is ``'example.com'`` and the `SERVER_NAME`
        in the wsgi `environ` is ``'staging.dev.example.com'`` the calculated
        subdomain will be ``'staging.dev'``.

        If the object passed as environ has an environ attribute, the value of
        this attribute is used instead.  This allows you to pass request
        objects.  Additionally `PATH_INFO` added as a default of the
        :class:`MapAdapter` so that you don't have to pass the path info to
        the match method.

        .. versionchanged:: 1.0.0
            If the passed server name specifies port 443, it will match
            if the incoming scheme is ``https`` without a port.

        .. versionchanged:: 1.0.0
            A warning is shown when the passed server name does not
            match the incoming WSGI server name.

        .. versionchanged:: 0.8
           This will no longer raise a ValueError when an unexpected server
           name was passed.

        .. versionchanged:: 0.5
            previously this method accepted a bogus `calculate_subdomain`
            parameter that did not have any effect.  It was removed because
            of that.

        :param environ: a WSGI environment.
        :param server_name: an optional server name hint (see above).
        :param subdomain: optionally the current subdomain (see above).
        """
        env = _get_environ(environ)
        wsgi_server_name = get_host(env).lower()
        scheme = env['wsgi.url_scheme']
        upgrade = any((v.strip() == 'upgrade' for v in env.get('HTTP_CONNECTION', '').lower().split(',')))
        if upgrade and env.get('HTTP_UPGRADE', '').lower() == 'websocket':
            scheme = 'wss' if scheme == 'https' else 'ws'
        if server_name is None:
            server_name = wsgi_server_name
        else:
            server_name = server_name.lower()
            if scheme in {'http', 'ws'} and server_name.endswith(':80'):
                server_name = server_name[:-3]
            elif scheme in {'https', 'wss'} and server_name.endswith(':443'):
                server_name = server_name[:-4]
        if subdomain is None and (not self.host_matching):
            cur_server_name = wsgi_server_name.split('.')
            real_server_name = server_name.split('.')
            offset = -len(real_server_name)
            if cur_server_name[offset:] != real_server_name:
                warnings.warn(f"Current server name {wsgi_server_name!r} doesn't match configured server name {server_name!r}", stacklevel=2)
                subdomain = '<invalid>'
            else:
                subdomain = '.'.join(filter(None, cur_server_name[:offset]))

        def _get_wsgi_string(name: str) -> str | None:
            val = env.get(name)
            if val is not None:
                return _wsgi_decoding_dance(val)
            return None
        script_name = _get_wsgi_string('SCRIPT_NAME')
        path_info = _get_wsgi_string('PATH_INFO')
        query_args = _get_wsgi_string('QUERY_STRING')
        return Map.bind(self, server_name, script_name, subdomain, scheme, env['REQUEST_METHOD'], path_info, query_args=query_args)

    def update(self) -> None:
        """Called before matching and building to keep the compiled rules
        in the correct order after things changed.
        """
        if not self._remap:
            return
        with self._remap_lock:
            if not self._remap:
                return
            self._matcher.update()
            for rules in self._rules_by_endpoint.values():
                rules.sort(key=lambda x: x.build_compare_key())
            self._remap = False

    def __repr__(self) -> str:
        rules = self.iter_rules()
        return f'{type(self).__name__}({pformat(list(rules))})'