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
def all_dep_attrs(self):
    try:
        attrs = [(self._res_name(), self._attr_path())]
    except Exception:
        attrs = []
    return itertools.chain(function.all_dep_attrs(self.args), attrs)