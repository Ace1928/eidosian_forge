import cgi
import datetime
import re
from simplegeneric import generic
from wsme.exc import ClientSideError, UnknownArgument, InvalidInput
from wsme.types import iscomplex, list_attributes, Unset
from wsme.types import UserType, ArrayType, DictType, File
from wsme.utils import parse_isodate, parse_isotime, parse_isodatetime
import wsme.runtime
def args_from_args(funcdef, args, kwargs):
    newargs = []
    for argdef, arg in zip(funcdef.arguments[:len(args)], args):
        try:
            newargs.append(from_param(argdef.datatype, arg))
        except Exception as e:
            if isinstance(argdef.datatype, UserType):
                datatype_name = argdef.datatype.name
            elif isinstance(argdef.datatype, type):
                datatype_name = argdef.datatype.__name__
            else:
                datatype_name = argdef.datatype.__class__.__name__
            raise InvalidInput(argdef.name, arg, 'unable to convert to %(datatype)s. Error: %(error)s' % {'datatype': datatype_name, 'error': e})
    newkwargs = {}
    for argname, value in kwargs.items():
        newkwargs[argname] = from_param(funcdef.get_arg(argname).datatype, value)
    return (newargs, newkwargs)