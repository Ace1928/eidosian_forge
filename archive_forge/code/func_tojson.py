import datetime
import decimal
import json
from simplegeneric import generic
import wsme.exc
import wsme.types
from wsme.types import Unset
import wsme.utils
@generic
def tojson(datatype, value):
    """
    A generic converter from python to jsonify-able datatypes.

    If a non-complex user specific type is to be used in the api,
    a specific tojson should be added::

        from wsme.protocol.restjson import tojson

        myspecialtype = object()

        @tojson.when_object(myspecialtype)
        def myspecialtype_tojson(datatype, value):
            return str(value)
    """
    if value is None:
        return None
    if wsme.types.iscomplex(datatype):
        d = dict()
        for attr in wsme.types.list_attributes(datatype):
            attr_value = getattr(value, attr.key)
            if attr_value is not Unset:
                d[attr.name] = tojson(attr.datatype, attr_value)
        return d
    elif wsme.types.isusertype(datatype):
        return tojson(datatype.basetype, datatype.tobasetype(value))
    return value