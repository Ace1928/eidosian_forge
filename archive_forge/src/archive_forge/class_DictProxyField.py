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
class DictProxyField(object):
    """Descriptor allowing us to assign pinning data as a dict of key_types

    This allows us to have an object field that will be a dict of key_type
    keys, allowing that will convert back to string-keyed dict.

    This will take care of the conversion while the dict field will make sure
    that we store the raw json-serializable data on the object.

    key_type should return a type that unambiguously responds to str
    so that calling key_type on it yields the same thing.
    """

    def __init__(self, dict_field_name, key_type=int):
        self._fld_name = dict_field_name
        self._key_type = key_type

    def __get__(self, obj, obj_type):
        if obj is None:
            return self
        if getattr(obj, self._fld_name) is None:
            return
        return {self._key_type(k): v for k, v in getattr(obj, self._fld_name).items()}

    def __set__(self, obj, val):
        if val is None:
            setattr(obj, self._fld_name, val)
        else:
            setattr(obj, self._fld_name, {str(k): v for k, v in val.items()})