import collections
from oslo_serialization import jsonutils
from heat.api.aws import utils as aws_utils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
class FindInMap(function.Function):
    """A function for resolving keys in the template mappings.

    Takes the form::

        { "Fn::FindInMap" : [ "mapping",
                              "key",
                              "value" ] }
    """

    def __init__(self, stack, fn_name, args):
        super(FindInMap, self).__init__(stack, fn_name, args)
        try:
            self._mapname, self._mapkey, self._mapvalue = self.args
        except ValueError as ex:
            raise KeyError(str(ex))

    def result(self):
        mapping = self.stack.t.maps[function.resolve(self._mapname)]
        key = function.resolve(self._mapkey)
        value = function.resolve(self._mapvalue)
        return mapping[key][value]