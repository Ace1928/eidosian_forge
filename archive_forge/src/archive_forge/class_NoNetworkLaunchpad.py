from contextlib import contextmanager
import launchpadlib
from launchpadlib.launchpad import Launchpad
from launchpadlib.credentials import (
class NoNetworkLaunchpad(Launchpad):
    """A Launchpad instance for tests with no network access.

    It's only useful for making sure that certain methods were called.
    It can't be used to interact with the API.
    """

    def __init__(self, credentials, authorization_engine, credential_store, service_root, cache, timeout, proxy_info, version):
        self.credentials = credentials
        self.authorization_engine = authorization_engine
        self.credential_store = credential_store
        self.passed_in_args = dict(service_root=service_root, cache=cache, timeout=timeout, proxy_info=proxy_info, version=version)

    @classmethod
    def authorization_engine_factory(cls, *args):
        return NoNetworkAuthorizationEngine(*args)