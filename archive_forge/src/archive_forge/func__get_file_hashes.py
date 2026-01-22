from collections.abc import Mapping
import hashlib
import queue
import string
import threading
import time
import typing as ty
import keystoneauth1
from keystoneauth1 import adapter as ks_adapter
from keystoneauth1 import discover
from openstack import _log
from openstack import exceptions
def _get_file_hashes(filename):
    _md5, _sha256 = (None, None)
    with open(filename, 'rb') as file_obj:
        _md5, _sha256 = _calculate_data_hashes(file_obj)
    return (_md5, _sha256)