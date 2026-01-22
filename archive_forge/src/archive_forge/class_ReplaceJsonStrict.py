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
class ReplaceJsonStrict(ReplaceJson):
    """A function for performing string substitutions.

    str_replace_strict is identical to the str_replace function, only
    a ValueError is raised if any of the params are not present in
    the template.
    """
    _strict = True