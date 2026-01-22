import datetime
import functools
import itertools
import uuid
from xmlrpc import client as xmlrpclib
import msgpack
from oslo_utils import importutils
def _create_default_registry():
    registry = HandlerRegistry()
    registry.register(DateTimeHandler(registry), reserved=True)
    registry.register(DateHandler(registry), reserved=True)
    registry.register(UUIDHandler(), reserved=True)
    registry.register(CountHandler(), reserved=True)
    registry.register(SetHandler(registry), reserved=True)
    registry.register(FrozenSetHandler(registry), reserved=True)
    if netaddr is not None:
        registry.register(NetAddrIPHandler(), reserved=True)
    registry.register(XMLRPCDateTimeHandler(registry), reserved=True)
    registry.frozen = True
    return registry