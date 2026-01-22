import requests
import warnings
from requests import adapters
from requests import sessions
from .. import exceptions as exc
from .._compat import gaecontrib
from .._compat import timeout
class _AppEnginePoolManager(object):
    """Implements urllib3's PoolManager API expected by requests.

    While a real PoolManager map hostnames to reusable Connections,
    AppEngine has no concept of a reusable connection to a host.
    So instead, this class constructs a small Connection per request,
    that is returned to the Adapter and used to access the URL.
    """

    def __init__(self, validate_certificate=True):
        self.appengine_manager = gaecontrib.AppEngineManager(validate_certificate=validate_certificate)

    def connection_from_url(self, url):
        return _AppEngineConnection(self.appengine_manager, url)

    def clear(self):
        pass