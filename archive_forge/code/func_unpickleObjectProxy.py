import os
import sys
import threading
import time
import traceback
import warnings
import weakref
import builtins
import pickle
import numpy as np
from ..util import cprint
def unpickleObjectProxy(processId, proxyId, typeStr, attributes=None, opts=None):
    if processId == os.getpid():
        obj = LocalObjectProxy.lookupProxyId(proxyId)
        if attributes is not None:
            for attr in attributes:
                obj = getattr(obj, attr)
        return obj
    else:
        proxy = ObjectProxy(processId, proxyId=proxyId, typeStr=typeStr)
        if opts is not None:
            proxy._setProxyOptions(**opts)
        return proxy