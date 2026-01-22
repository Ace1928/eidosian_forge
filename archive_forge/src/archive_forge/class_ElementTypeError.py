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
class ElementTypeError(TypeError):

    def __init__(self, expected, key, value):
        super(ElementTypeError, self).__init__(_('Element %(key)s:%(val)s must be of type %(expected)s not %(actual)s') % {'key': key, 'val': repr(value), 'expected': expected, 'actual': value.__class__.__name__})