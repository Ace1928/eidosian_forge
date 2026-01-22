import abc
from collections import abc as collections_abc
import datetime
from distutils import versionpredicate
import re
import uuid
import warnings
import copy
import iso8601
import netaddr
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_versionedobjects._i18n import _
from oslo_versionedobjects import _utils
from oslo_versionedobjects import exception
class PCIAddress(StringPattern):
    PATTERN = '^[0-9a-f]{4}:[0-9a-f]{2}:[0-1][0-9a-f].[0-7]$'
    _REGEX = re.compile(PATTERN)

    @staticmethod
    def coerce(obj, attr, value):
        if isinstance(value, str):
            newvalue = value.lower()
            if PCIAddress._REGEX.match(newvalue):
                return newvalue
        raise ValueError(_('Malformed PCI address %s') % (value,))