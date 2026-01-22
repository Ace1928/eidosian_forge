import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class KrbAsRep(KerberosV5Msg):
    """The KRB_AS_REP message.

    The KRB_AS_REP message is used for a reply from the KDC to a KRB_AS_REQ message. The KRB_TGS_REP message is
    identical except for the tag and msg-type.

    The ASN.1 definition for the KDC-REP structure is defined in `RFC 4120 5.4.2`_::

        KDC-REP         ::= SEQUENCE {
            pvno            [0] INTEGER (5),
            msg-type        [1] INTEGER (11 -- AS -- | 13 -- TGS --),
            padata          [2] SEQUENCE OF PA-DATA OPTIONAL
                                -- NOTE: not empty --,
            crealm          [3] Realm,
            cname           [4] PrincipalName,
            ticket          [5] Ticket,
            enc-part        [6] EncryptedData
                                -- EncASRepPart or EncTGSRepPart,
                                -- as appropriate
        }

    Args:
        sequence: The ASN.1 sequence value as a dict to unpack.

    Attributes:
        padata (PAData): The pre-authentication data.
        crealm (bytes): The client realm.
        cname (PrincipalName): The client principal name.
        ticket (Ticket): The newly issued ticket.
        enc_part (EncryptedData): The encrypted part of the message.

    .. _RFC 4120 5.4.2:
        https://www.rfc-editor.org/rfc/rfc4120#section-5.4.2
    """
    MESSAGE_TYPE = KerberosMessageType.as_rep
    PARSE_MAP = [('pvno', 'PVNO', ParseType.default), ('msg-type', 'MESSAGE_TYPE', ParseType.enum), ('padata', 'padata', ParseType.token), ('crealm', 'crealm', ParseType.text), ('cname', 'cname', ParseType.principal_name), ('ticket', 'ticket', ParseType.token), ('enc-part', 'enc_part', ParseType.token)]

    def __init__(self, sequence: typing.Dict[int, ASN1Value]) -> None:

        def unpack_padata(value: typing.Union[ASN1Value, bytes]) -> typing.List:
            return [PAData.unpack(p) for p in unpack_asn1_sequence(value)]
        self.padata = get_sequence_value(sequence, 2, 'KDC-REP', 'pa-data', unpack_padata)
        self.crealm = get_sequence_value(sequence, 3, 'KDC-REP', 'crealm', unpack_asn1_general_string)
        self.cname = get_sequence_value(sequence, 4, 'KDC-REP', 'cname', unpack_principal_name)
        self.ticket = get_sequence_value(sequence, 5, 'KDC-REP', 'ticket', Ticket.unpack)
        self.enc_part = get_sequence_value(sequence, 6, 'KDC-REP', 'enc-part', EncryptedData.unpack)