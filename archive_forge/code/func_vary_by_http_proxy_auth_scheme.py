import io
import socket
import sys
import threading
from http.client import UnknownProtocol, parse_headers
from http.server import SimpleHTTPRequestHandler
import breezy
from .. import (config, controldir, debug, errors, osutils, tests, trace,
from ..bzr import remote as _mod_remote
from ..transport import remote
from ..transport.http import urllib
from ..transport.http.urllib import (AbstractAuthHandler, BasicAuthHandler,
from . import features, http_server, http_utils, test_server
from .scenarios import load_tests_apply_scenarios, multiply_scenarios
def vary_by_http_proxy_auth_scheme():
    scenarios = [('proxy-basic', dict(_auth_server=http_utils.ProxyBasicAuthServer)), ('proxy-digest', dict(_auth_server=http_utils.ProxyDigestAuthServer)), ('proxy-basicdigest', dict(_auth_server=http_utils.ProxyBasicAndDigestAuthServer))]
    for scenario_id, scenario_dict in scenarios:
        scenario_dict.update(_auth_header='Proxy-Authorization', _username_prompt_prefix='Proxy ', _password_prompt_prefix='Proxy ')
    return scenarios