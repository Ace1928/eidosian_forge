import functools
from keystoneauth1 import _utils as utils
from keystoneauth1.access import service_catalog
from keystoneauth1.access import service_providers
@property
def trust_scoped(self):
    try:
        return bool(self._trust)
    except KeyError:
        return False