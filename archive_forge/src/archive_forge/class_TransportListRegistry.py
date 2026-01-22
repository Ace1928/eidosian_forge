import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
class TransportListRegistry(registry.Registry):
    """A registry which simplifies tracking available Transports.

    A registration of a new protocol requires two steps:
    1) register the prefix with the function register_transport( )
    2) register the protocol provider with the function
    register_transport_provider( ) ( and the "lazy" variant )

    This is needed because:
    a) a single provider can support multiple protocols (like the ftp
    provider which supports both the ftp:// and the aftp:// protocols)
    b) a single protocol can have multiple providers (like the http://
    protocol which was supported by both the urllib and pycurl providers)
    """

    def register_transport_provider(self, key, obj):
        self.get(key).insert(0, registry._ObjectGetter(obj))

    def register_lazy_transport_provider(self, key, module_name, member_name):
        self.get(key).insert(0, registry._LazyObjectGetter(module_name, member_name))

    def register_transport(self, key, help=None):
        self.register(key, [], help)