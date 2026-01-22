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
def _coerce_element(self, element):
    if hasattr(self, '_element_type') and self._element_type is not None:
        return self._element_type.coerce(self._obj, '%s[%s]' % (self._field, element), element)
    else:
        return element