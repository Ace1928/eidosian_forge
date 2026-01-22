import functools
import weakref
from .__wrapt__ import ObjectProxy, _FunctionWrapperBase
def _weak_function_proxy_callback(ref, proxy, callback):
    if proxy._self_expired:
        return
    proxy._self_expired = True
    if callback is not None:
        callback(proxy)