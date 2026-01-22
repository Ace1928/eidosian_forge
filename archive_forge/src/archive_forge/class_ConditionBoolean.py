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
class ConditionBoolean(function.Function):
    """Abstract parent class of boolean condition functions."""

    def __init__(self, stack, fn_name, args):
        super(ConditionBoolean, self).__init__(stack, fn_name, args)
        self._check_args()

    def _check_args(self):
        if not (isinstance(self.args, collections.abc.Sequence) and (not isinstance(self.args, str))):
            msg = _('Arguments to "%s" must be a list of conditions')
            raise ValueError(msg % self.fn_name)
        if not self.args or len(self.args) < 2:
            msg = _('The minimum number of condition arguments to "%s" is 2.')
            raise ValueError(msg % self.fn_name)

    def _get_condition(self, arg):
        if isinstance(arg, bool):
            return arg
        conditions = self.stack.t.conditions(self.stack)
        return conditions.is_enabled(arg)