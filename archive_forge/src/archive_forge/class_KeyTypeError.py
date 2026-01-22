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
class KeyTypeError(TypeError):

    def __init__(self, expected, value):
        super(KeyTypeError, self).__init__(_('Key %(key)s must be of type %(expected)s not %(actual)s') % {'key': repr(value), 'expected': expected.__name__, 'actual': value.__class__.__name__})