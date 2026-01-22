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
class ListConcat(function.Function):
    """A function for extending lists.

    Takes the form::

        list_concat:
          - [<value 1>, <value 2>]
          - [<value 3>, <value 4>]

    And resolves to::

        [<value 1>, <value 2>, <value 3>, <value 4>]

    """
    _unique = False

    def __init__(self, stack, fn_name, args):
        super(ListConcat, self).__init__(stack, fn_name, args)
        example = _('"%s" : [ [ <value 1>, <value 2> ], [ <value 3>, <value 4> ] ]') % fn_name
        self.fmt_data = {'fn_name': fn_name, 'example': example}

    def result(self):
        args = function.resolve(self.args)
        if isinstance(args, str) or not isinstance(args, collections.abc.Sequence):
            raise TypeError(_('Incorrect arguments to "%(fn_name)s" should be: %(example)s') % self.fmt_data)

        def ensure_list(m):
            if m is None:
                return []
            elif isinstance(m, collections.abc.Sequence) and (not isinstance(m, str)):
                return m
            else:
                msg = _('Incorrect arguments: Items to concat must be lists. %(args)s contains an item that is not a list: %(item)s')
                raise TypeError(msg % dict(item=jsonutils.dumps(m), args=jsonutils.dumps(args)))
        ret_list = []
        for m in args:
            ret_list.extend(ensure_list(m))
        if not self._unique:
            return ret_list
        unique_list = []
        for item in ret_list:
            if item not in unique_list:
                unique_list.append(item)
        return unique_list