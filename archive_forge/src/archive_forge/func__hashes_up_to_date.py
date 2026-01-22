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
def _hashes_up_to_date(md5, sha256, md5_key, sha256_key):
    """Compare md5 and sha256 hashes for being up to date

    md5 and sha256 are the current values.
    md5_key and sha256_key are the previous values.
    """
    up_to_date = False
    if md5 and md5_key == md5:
        up_to_date = True
    if sha256 and sha256_key == sha256:
        up_to_date = True
    if md5 and md5_key != md5:
        up_to_date = False
    if sha256 and sha256_key != sha256:
        up_to_date = False
    return up_to_date