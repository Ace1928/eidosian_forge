import base64
import calendar
from ipaddress import AddressValueError
from ipaddress import IPv4Address
from ipaddress import IPv6Address
import re
import struct
import time
from urllib.parse import urlparse
from saml2 import time_util
def valid_instance(instance):
    instclass = instance.__class__
    class_name = instclass.__name__
    if instclass.c_value_type and instance.text:
        try:
            validate_value_type(instance.text.strip(), instclass.c_value_type)
        except NotValid as exc:
            raise NotValid(f"Class '{class_name}' instance: {exc.args[0]}")
    for name, typ, required in instclass.c_attributes.values():
        value = getattr(instance, name, '')
        if required and (not value):
            txt = f"Required value on property '{name}' missing"
            raise MustValueError(f"Class '{class_name}' instance: {txt}")
        if value:
            try:
                if isinstance(typ, type):
                    if typ.c_value_type:
                        spec = typ.c_value_type
                    else:
                        spec = {'base': 'string'}
                    validate_value_type(value, spec)
                else:
                    valid(typ, value)
            except (NotValid, ValueError) as exc:
                txt = ERROR_TEXT % (value, name, exc.args[0])
                raise NotValid(f"Class '{class_name}' instance: {txt}")
    for name, _spec in instclass.c_children.values():
        value = getattr(instance, name, '')
        try:
            _card = instclass.c_cardinality[name]
            try:
                _cmin = _card['min']
            except KeyError:
                _cmin = None
            try:
                _cmax = _card['max']
            except KeyError:
                _cmax = None
        except KeyError:
            _cmin = _cmax = _card = None
        if value:
            if isinstance(value, list):
                _list = True
                vlen = len(value)
            else:
                _list = False
                vlen = 1
            if _card:
                if _cmin is not None and _cmin > vlen:
                    raise NotValid(f"Class '{class_name}' instance cardinality error: less then min ({vlen}<{_cmin})")
                if _cmax is not None and vlen > _cmax:
                    raise NotValid(f"Class '{class_name}' instance cardinality error: more then max ({vlen}>{_cmax})")
            if _list:
                for val in value:
                    _valid_instance(instance, val)
            else:
                _valid_instance(instance, value)
        elif _cmin:
            raise NotValid(f"Class '{class_name}' instance cardinality error: too few values on {name}")
    return True