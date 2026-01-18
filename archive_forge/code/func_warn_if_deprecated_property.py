import abc
import collections
import inspect
import itertools
import operator
import typing as ty
import urllib.parse
import warnings
import jsonpatch
from keystoneauth1 import adapter
from keystoneauth1 import discover
from requests import structures
from openstack import _log
from openstack import exceptions
from openstack import format
from openstack import utils
from openstack import warnings as os_warnings
def warn_if_deprecated_property(self, value):
    deprecated = object.__getattribute__(self, 'deprecated')
    deprecation_reason = object.__getattribute__(self, 'deprecation_reason')
    if value and deprecated:
        warnings.warn('The field %r has been deprecated. %s' % (self.name, deprecation_reason or 'Avoid usage.'), os_warnings.RemovedFieldWarning)
    return value