import cgi
import datetime
import re
from simplegeneric import generic
from wsme.exc import ClientSideError, UnknownArgument, InvalidInput
from wsme.types import iscomplex, list_attributes, Unset
from wsme.types import UserType, ArrayType, DictType, File
from wsme.utils import parse_isodate, parse_isotime, parse_isodatetime
import wsme.runtime
@from_param.when_type(UserType)
def usertype_from_param(datatype, value):
    return datatype.frombasetype(from_param(datatype.basetype, value))