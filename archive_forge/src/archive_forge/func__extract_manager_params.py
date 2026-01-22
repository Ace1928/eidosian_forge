from ncclient import operations
from ncclient import transport
import socket
import logging
import functools
from ncclient.xml_ import *
def _extract_manager_params(kwds):
    manager_params = kwds.pop('manager_params', {})
    if 'timeout' not in manager_params and 'timeout' in kwds:
        manager_params['timeout'] = kwds['timeout']
    return manager_params