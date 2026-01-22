import collections
from oslo_serialization import jsonutils
from heat.api.aws import utils as aws_utils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
def Ref(stack, fn_name, args):
    """A function for resolving parameters or resource references.

    Takes the form::

        { "Ref" : "<param_name>" }

    or::

        { "Ref" : "<resource_name>" }
    """
    if stack is None or args in stack:
        RefClass = hot_funcs.GetResource
    else:
        RefClass = ParamRef
    return RefClass(stack, fn_name, args)