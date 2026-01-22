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
class Removed(function.Function):
    """This function existed in previous versions of HOT, but has been removed.

    Check the HOT guide for an equivalent native function.
    """

    def validate(self):
        exp = _('The function %s is not supported in this version of HOT.') % self.fn_name
        raise exception.InvalidTemplateVersion(explanation=exp)

    def result(self):
        return super(Removed, self).result()