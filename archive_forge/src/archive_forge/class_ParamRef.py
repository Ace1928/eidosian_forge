import collections
from oslo_serialization import jsonutils
from heat.api.aws import utils as aws_utils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
class ParamRef(function.Function):
    """A function for resolving parameter references.

    Takes the form::

        { "Ref" : "<param_name>" }
    """

    def __init__(self, stack, fn_name, args):
        super(ParamRef, self).__init__(stack, fn_name, args)
        self.parameters = self.stack.parameters

    def result(self):
        param_name = function.resolve(self.args)
        try:
            return self.parameters[param_name]
        except KeyError:
            raise exception.InvalidTemplateReference(resource=param_name, key='unknown')