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
class IfNullable(If):
    """A function to return corresponding value based on condition evaluation.

    Takes the form::

        if:
          - <condition_name>
          - <value_if_true>
          - <value_if_false>

    The value_if_true to be returned if the specified condition evaluates
    to true, the value_if_false to be returned if the specified condition
    evaluates to false.

    If the value_if_false is omitted and the condition is false, the enclosing
    item (list item, dictionary key/value pair, property definition) will be
    treated as if it were not mentioned in the template::

        if:
          - <condition_name>
          - <value_if_true>
    """

    def _read_args(self):
        if not 2 <= len(self.args) <= 3:
            raise ValueError()
        if len(self.args) == 2:
            condition, value_if_true = self.args
            return (condition, value_if_true, Ellipsis)
        return self.args