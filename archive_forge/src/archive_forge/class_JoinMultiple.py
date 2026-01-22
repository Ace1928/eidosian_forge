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
class JoinMultiple(function.Function):
    """A function for joining one or more lists of strings.

    Takes the form::

        list_join:
          - <delim>
          - - <string_1>
            - <string_2>
            - ...
          - - ...

    And resolves to::

        "<string_1><delim><string_2><delim>..."

    Optionally multiple lists may be specified, which will also be joined.
    """

    def __init__(self, stack, fn_name, args):
        super(JoinMultiple, self).__init__(stack, fn_name, args)
        example = '"%s" : [ " ", [ "str1", "str2"] ...]' % fn_name
        fmt_data = {'fn_name': fn_name, 'example': example}
        if not isinstance(args, list):
            raise TypeError(_('Incorrect arguments to "%(fn_name)s" should be: %(example)s') % fmt_data)
        try:
            self._delim = args[0]
            self._joinlists = args[1:]
            if len(self._joinlists) < 1:
                raise ValueError
        except (IndexError, ValueError):
            raise ValueError(_('Incorrect arguments to "%(fn_name)s" should be: %(example)s') % fmt_data)

    def result(self):
        r_joinlists = function.resolve(self._joinlists)
        strings = []
        for jl in r_joinlists:
            if jl:
                if isinstance(jl, str) or not isinstance(jl, collections.abc.Sequence):
                    raise TypeError(_('"%s" must operate on a list') % self.fn_name)
                strings += jl
        delim = function.resolve(self._delim)
        if not isinstance(delim, str):
            raise TypeError(_('"%s" delimiter must be a string') % self.fn_name)

        def ensure_string(s):
            msg = _('Items to join must be string, map or list not %s') % repr(s)[:200]
            if s is None:
                return ''
            elif isinstance(s, str):
                return s
            elif isinstance(s, (collections.abc.Mapping, collections.abc.Sequence)):
                try:
                    return jsonutils.dumps(s, default=None, sort_keys=True)
                except TypeError:
                    msg = _('Items to join must be string, map or list. %s failed json serialization') % repr(s)[:200]
            raise TypeError(msg)
        return delim.join((ensure_string(s) for s in strings))