import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class KrbError(KerberosV5Msg):
    """The KRB_ERROR message.

    The KRB_ERROR is a message sent in the occurrence of an error.

    The ASN.1 definition for the KRB_ERROR structure is defined in `RFC 4120 5.9.1`_::

        KRB-ERROR       ::= [APPLICATION 30] SEQUENCE {
            pvno            [0] INTEGER (5),
            msg-type        [1] INTEGER (30),
            ctime           [2] KerberosTime OPTIONAL,
            cusec           [3] Microseconds OPTIONAL,
            stime           [4] KerberosTime,
            susec           [5] Microseconds,
            error-code      [6] Int32,
            crealm          [7] Realm OPTIONAL,
            cname           [8] PrincipalName OPTIONAL,
            realm           [9] Realm -- service realm --,
            sname           [10] PrincipalName -- service name --,
            e-text          [11] KerberosString OPTIONAL,
            e-data          [12] OCTET STRING OPTIONAL
        }

    Args:
        sequence: The ASN.1 sequence value as a dict to unpack.

    Attributes:
        ctime (datetime.datetime): The current time on the client's host.
        cusec (int): The microsecond part of the client's timestamp.
        stime (datetime.datetime): The current time of the server.
        susec (int): The microsecond part of the server's timestamp.
        error_code (KerberosErrorCode): THe error code returned by the kerberos when a request fails.
        crealm (bytes): The realm that issues a ticket.
        cname (PrincipalName): The principal name in the ticket.
        realm (bytes): The service realm.
        sname (PrincipalName): The service name.
        e_text (bytes): Additional text to explain the error code.
        e_data (bytes): Additional data about the error.

    .. _RFC 4120 5.9.1:
        https://www.rfc-editor.org/rfc/rfc4120#section-5.9.1
    """
    MESSAGE_TYPE = KerberosMessageType.error
    PARSE_MAP = [('pvno', 'PVNO', ParseType.default), ('msg-type', 'MESSAGE_TYPE', ParseType.enum), ('ctime', 'ctime', ParseType.datetime), ('cusec', 'cusec', ParseType.default), ('stime', 'stime', ParseType.datetime), ('susec', 'susec', ParseType.default), ('error-code', 'error_code', ParseType.enum), ('crealm', 'crealm', ParseType.text), ('cname', 'cname', ParseType.principal_name), ('realm', 'realm', ParseType.text), ('sname', 'sname', ParseType.principal_name), ('e-text', 'e_text', ParseType.text), ('e-data', 'e_data', ParseType.bytes)]

    def __init__(self, sequence: typing.Dict[int, ASN1Value]) -> None:
        self.ctime = get_sequence_value(sequence, 2, 'KRB-ERROR', 'ctime', unpack_asn1_generalized_time)
        self.cusec = get_sequence_value(sequence, 3, 'KRB-ERROR', 'cusec', unpack_asn1_integer)
        self.stime = get_sequence_value(sequence, 4, 'KRB-ERROR', 'stime', unpack_asn1_generalized_time)
        self.susec = get_sequence_value(sequence, 5, 'KRB-ERROR', 'susec', unpack_asn1_integer)
        self.error_code = KerberosErrorCode(get_sequence_value(sequence, 6, 'KRB-ERROR', 'error-code', unpack_asn1_integer))
        self.crealm = get_sequence_value(sequence, 7, 'KRB-ERROR', 'crealm', unpack_asn1_general_string)
        self.cname = get_sequence_value(sequence, 8, 'KRB-ERROR', 'cname', unpack_principal_name)
        self.realm = get_sequence_value(sequence, 9, 'KRB-ERROR', 'realm', unpack_asn1_general_string)
        self.sname = get_sequence_value(sequence, 10, 'KRB-ERROR', 'realm', unpack_principal_name)
        self.e_text = get_sequence_value(sequence, 11, 'KRB-ERROR', 'e-text', unpack_asn1_general_string)
        self.e_data = get_sequence_value(sequence, 12, 'KRB-ERROR', 'e-data', unpack_asn1_octet_string)