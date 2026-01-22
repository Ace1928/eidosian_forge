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
class DeferredObjectProxy(ObjectProxy):
    """
    This class represents an attribute (or sub-attribute) of a proxied object.
    It is used to speed up attribute requests. Take the following scenario::
    
        rsys = proc._import('sys')
        rsys.stdout.write('hello')
        
    For this simple example, a total of 4 synchronous requests are made to 
    the remote process: 
    
    1) import sys
    2) getattr(sys, 'stdout')
    3) getattr(stdout, 'write')
    4) write('hello')
    
    This takes a lot longer than running the equivalent code locally. To
    speed things up, we can 'defer' the two attribute lookups so they are
    only carried out when neccessary::
    
        rsys = proc._import('sys')
        rsys._setProxyOptions(deferGetattr=True)
        rsys.stdout.write('hello')
        
    This example only makes two requests to the remote process; the two 
    attribute lookups immediately return DeferredObjectProxy instances 
    immediately without contacting the remote process. When the call 
    to write() is made, all attribute requests are processed at the same time.
    
    Note that if the attributes requested do not exist on the remote object, 
    making the call to write() will raise an AttributeError.
    """

    def __init__(self, parentProxy, attribute):
        for k in ['_processId', '_typeStr', '_proxyId', '_handler']:
            self.__dict__[k] = getattr(parentProxy, k)
        self.__dict__['_parent'] = parentProxy
        self.__dict__['_attributes'] = parentProxy._attributes + (attribute,)
        self.__dict__['_proxyOptions'] = parentProxy._proxyOptions.copy()

    def __repr__(self):
        return ObjectProxy.__repr__(self) + '.' + '.'.join(self._attributes)

    def _undefer(self):
        """
        Return a non-deferred ObjectProxy referencing the same object
        """
        return self._parent.__getattr__(self._attributes[-1], _deferGetattr=False)