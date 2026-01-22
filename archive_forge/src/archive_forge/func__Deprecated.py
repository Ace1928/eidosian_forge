import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
def _Deprecated(name):
    if _Deprecated.count > 0:
        _Deprecated.count -= 1
        warnings.warn('Call to deprecated create function %s(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.' % name, category=DeprecationWarning, stacklevel=3)