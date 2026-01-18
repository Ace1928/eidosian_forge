import ctypes
import inspect
from pkg_resources import parse_version
import textwrap
import time
import types
import eventlet
from eventlet import tpool
import netaddr
from oslo_concurrency import lockutils
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
import six
from os_win import constants
from os_win import exceptions
def wmi_retry_decorator_hresult(exceptions=exceptions.x_wmi, **kwargs):
    """Retry decorator that can be used for specific WMI HRESULTs"""

    def err_code_func(exc):
        com_error = getattr(exc, 'com_error', None)
        if com_error:
            return get_com_error_hresult(com_error)
    return retry_decorator(extract_err_code_func=err_code_func, exceptions=exceptions, **kwargs)