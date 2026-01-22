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
@staticmethod
def _get_all_obj_names(obj):
    obj_names = []
    for parent in obj.__class__.mro():
        if not hasattr(parent, 'obj_name'):
            continue
        obj_names.append(parent.obj_name())
    return obj_names