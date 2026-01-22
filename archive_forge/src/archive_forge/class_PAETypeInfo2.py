import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class PAETypeInfo2:
    """Kerberos PA-ETYPE-INFO2 container.

    The ASN.1 definition for the PA-ETYPE-INFO2 structure is defined in `RFC 4120 5.2.7.5`_::

        ETYPE-INFO2-ENTRY       ::= SEQUENCE {
            etype           [0] Int32,
            salt            [1] KerberosString OPTIONAL,
            s2kparams       [2] OCTET STRING OPTIONAL
        }

    Args:
        etype: The etype that defines the cipher used.
        salt: The used in the cipher associated with the cryptosystem.
        s2kparams: Extra params to be interpreted by the cipher associated with the cryptosystem.

    Attributes:
        etype (KerberosEncryptionType): See args.
        salt (Optional[bytes]): See args.
        s2kparams (Optional[bytes]): See args.

    .. RFC 4120 5.2.7.5:
        https://www.rfc-editor.org/rfc/rfc4120#section-5.2.7.5
    """
    PARSE_MAP = [('etype', 'etype', ParseType.enum), ('salt', 'salt', ParseType.bytes), ('s2kparams', 's2kparams', ParseType.bytes)]

    def __init__(self, etype: KerberosEncryptionType, salt: typing.Optional[bytes], s2kparams: typing.Optional[bytes]) -> None:
        self.etype = etype
        self.salt = salt
        self.s2kparams = s2kparams

    @staticmethod
    def unpack(value: typing.Union[ASN1Value, bytes]) -> 'PAETypeInfo2':
        sequence = unpack_asn1_tagged_sequence(value)
        etype = KerberosEncryptionType(get_sequence_value(sequence, 0, 'PA-ETYPE-INFO2', 'etype', unpack_asn1_integer))
        salt = get_sequence_value(sequence, 1, 'ETYPE-INFO2-ENTRY', 'salt', unpack_asn1_general_string)
        s2kparams = get_sequence_value(sequence, 2, 'ETYPE-INFO2-ENTRY', 's2kparams', unpack_asn1_octet_string)
        return PAETypeInfo2(etype, salt, s2kparams)