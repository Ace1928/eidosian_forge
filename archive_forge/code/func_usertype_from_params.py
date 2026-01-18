import cgi
import datetime
import re
from simplegeneric import generic
from wsme.exc import ClientSideError, UnknownArgument, InvalidInput
from wsme.types import iscomplex, list_attributes, Unset
from wsme.types import UserType, ArrayType, DictType, File
from wsme.utils import parse_isodate, parse_isotime, parse_isodatetime
import wsme.runtime
@from_params.when_type(UserType)
def usertype_from_params(datatype, params, path, hit_paths):
    value = from_params(datatype.basetype, params, path, hit_paths)
    if value is not Unset:
        return datatype.frombasetype(value)
    return Unset