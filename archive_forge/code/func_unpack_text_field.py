import typing
from spnego._asn1 import (
def unpack_text_field(sequence: typing.Dict[int, ASN1Value], idx: int, structure: str, name: str, **kwargs: typing.Optional[str]) -> typing.Optional[str]:
    """Extracts a text field from a tagged ASN.1 sequence."""
    raw_value = get_sequence_value(sequence, idx, structure, name, unpack_asn1_octet_string)
    if raw_value is None:
        if 'default' not in kwargs:
            raise ValueError("Missing mandatory text field '%s' in '%s'" % (name, structure))
        return kwargs['default']
    return raw_value.decode('utf-16-le')