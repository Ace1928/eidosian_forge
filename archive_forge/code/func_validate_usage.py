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
def validate_usage(self, args):
    if not (isinstance(args, list) and all([isinstance(a, str) for a in args])):
        msg = _('Argument to function "%s" must be a list of strings')
        raise TypeError(msg % self.fn_name)
    if len(args) != 2:
        msg = _('Function "%s" usage: ["<algorithm>", "<value>"]')
        raise ValueError(msg % self.fn_name)
    algorithms = hashlib.algorithms_available
    if args[0].lower() not in algorithms:
        msg = _('Algorithm must be one of %s')
        raise ValueError(msg % str(algorithms))