import collections
import functools
import hashlib
import itertools
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from urllib import parse as urlparse
import yaql
from yaql.language import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import function
class GetAtt(GetAttThenSelect):
    """A function for resolving resource attributes.

    Takes the form::

        get_attr:
          - <resource_name>
          - <attribute_name>
          - <path1>
          - ...
    """

    def result(self):
        path_components = function.resolve(self._path_components)
        attribute = function.resolve(self._attribute)
        resource = self._resource()
        if self._result_ready(resource):
            return resource.FnGetAtt(attribute, *path_components)
        else:
            return None

    def _attr_path(self):
        path = function.resolve(self._path_components)
        attr = function.resolve(self._attribute)
        if path:
            return tuple([attr] + path)
        else:
            return attr