import functools
import logging
import re
import warnings
import manilaclient
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient import utils
def add_versioned_method(versioned_method):
    _VERSIONED_METHOD_MAP.setdefault(versioned_method.name, [])
    _VERSIONED_METHOD_MAP[versioned_method.name].append(versioned_method)