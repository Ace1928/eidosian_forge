import calendar
import datetime
import functools
from base64 import b16encode
from functools import partial
from os import PathLike
from typing import (
from cryptography import utils, x509
from cryptography.hazmat.primitives.asymmetric import (
from OpenSSL._util import (
def _get_asn1_time(timestamp: Any) -> Optional[bytes]:
    """
    Retrieve the time value of an ASN1 time object.

    @param timestamp: An ASN1_GENERALIZEDTIME* (or an object safely castable to
        that type) from which the time value will be retrieved.

    @return: The time value from C{timestamp} as a L{bytes} string in a certain
        format.  Or C{None} if the object contains no time value.
    """
    string_timestamp = _ffi.cast('ASN1_STRING*', timestamp)
    if _lib.ASN1_STRING_length(string_timestamp) == 0:
        return None
    elif _lib.ASN1_STRING_type(string_timestamp) == _lib.V_ASN1_GENERALIZEDTIME:
        return _ffi.string(_lib.ASN1_STRING_get0_data(string_timestamp))
    else:
        generalized_timestamp = _ffi.new('ASN1_GENERALIZEDTIME**')
        _lib.ASN1_TIME_to_generalizedtime(timestamp, generalized_timestamp)
        if generalized_timestamp[0] == _ffi.NULL:
            _untested_error('ASN1_TIME_to_generalizedtime')
        else:
            string_timestamp = _ffi.cast('ASN1_STRING*', generalized_timestamp[0])
            string_data = _lib.ASN1_STRING_get0_data(string_timestamp)
            string_result = _ffi.string(string_data)
            _lib.ASN1_GENERALIZEDTIME_free(generalized_timestamp[0])
            return string_result