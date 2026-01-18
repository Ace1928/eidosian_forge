import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def unpack_asn1_generalized_time(value: typing.Union[ASN1Value, bytes]) -> datetime.datetime:
    """Unpacks an ASN.1 GeneralizedTime value."""
    data = to_text(extract_asn1_tlv(value, TagClass.universal, TypeTagNumber.generalized_time))
    if data.endswith('Z'):
        data = data[:-1]
    err = None
    for datetime_format in ['%Y%m%d%H%M%S.%f', '%Y%m%d%H%M%S']:
        try:
            dt = datetime.datetime.strptime(data, datetime_format)
            return dt.replace(tzinfo=datetime.timezone.utc)
        except ValueError as e:
            err = e
    else:
        raise err