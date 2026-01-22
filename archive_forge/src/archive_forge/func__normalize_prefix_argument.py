import errno
import hashlib
import os.path  # pylint: disable-msg=W0404
import warnings
from typing import Dict, List, Type, Iterator, Optional
from os.path import join as pjoin
import libcloud.utils.files
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import BaseDriver, Connection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError
from libcloud.storage.types import ObjectDoesNotExistError
def _normalize_prefix_argument(self, prefix, ex_prefix):
    if ex_prefix:
        warnings.warn('The ``ex_prefix`` argument is deprecated - please update code to use ``prefix``', DeprecationWarning)
        return ex_prefix
    return prefix