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
class GetParam(function.Function):
    """A function for resolving parameter references.

    Takes the form::

        get_param: <param_name>

    or::

        get_param:
          - <param_name>
          - <path1>
          - ...
    """

    def __init__(self, stack, fn_name, args):
        super(GetParam, self).__init__(stack, fn_name, args)
        if self.stack is not None:
            self.parameters = self.stack.parameters
        else:
            self.parameters = None

    def result(self):
        assert self.parameters is not None, 'No stack definition in Function'
        args = function.resolve(self.args)
        if not args:
            raise ValueError(_('Function "%s" must have arguments') % self.fn_name)
        if isinstance(args, str):
            param_name = args
            path_components = []
        elif isinstance(args, collections.abc.Sequence):
            param_name = args[0]
            path_components = args[1:]
        else:
            raise TypeError(_('Argument to "%s" must be string or list') % self.fn_name)
        if not isinstance(param_name, str):
            raise TypeError(_('Parameter name in "%s" must be string') % self.fn_name)
        try:
            parameter = self.parameters[param_name]
        except KeyError:
            raise exception.UserParameterMissing(key=param_name)

        def get_path_component(collection, key):
            if not isinstance(collection, (collections.abc.Mapping, collections.abc.Sequence)):
                raise TypeError(_('"%s" can\'t traverse path') % self.fn_name)
            if not isinstance(key, (str, int)):
                raise TypeError(_('Path components in "%s" must be strings') % self.fn_name)
            if isinstance(collection, collections.abc.Sequence) and isinstance(key, str):
                try:
                    key = int(key)
                except ValueError:
                    raise TypeError(_("Path components in '%s' must be a string that can be parsed into an integer.") % self.fn_name)
            return collection[key]
        try:
            return functools.reduce(get_path_component, path_components, parameter)
        except (KeyError, IndexError, TypeError):
            return ''