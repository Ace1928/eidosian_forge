import re
import sys
import six
from six.moves.urllib import parse as urlparse
from routes.util import _url_quote as url_quote, _str_encode, as_unicode
def _setup_route(self):
    self.routelist = routelist = self._pathkeys(self.routepath)
    routekeys = frozenset((key['name'] for key in routelist if isinstance(key, dict)))
    self.dotkeys = frozenset((key['name'] for key in routelist if isinstance(key, dict) and key['type'] == '.'))
    if not self.minimization:
        self.make_full_route()
    self.req_regs = {}
    for key, val in six.iteritems(self.reqs):
        self.req_regs[key] = re.compile('^' + val + '$')
    self.defaults, defaultkeys = self._defaults(routekeys, self.reserved_keys, self._kargs.copy())
    self.maxkeys = defaultkeys | routekeys
    self.minkeys, self.routebackwards = self._minkeys(routelist[:])
    self.hardcoded = frozenset((key for key in self.maxkeys if key not in routekeys and self.defaults[key] is not None))
    self._default_keys = frozenset(self.defaults.keys())