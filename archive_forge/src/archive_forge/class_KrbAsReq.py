import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class KrbAsReq(KerberosV5Msg):
    """The KRB_AS_REQ message.

    The KRB_AS_REQ message is used when the client wishes to retrieve a the initial ticket for a service. The
    KRB_TGS_REQ message is identical except for the tag and msg-type is used when retrieving additional tickets for a
    service.

    The ASN.1 definition for the KDC-REQ structure is defined in `RFC 4120 5.4.1`_::

        KDC-REQ         ::= SEQUENCE {
            -- NOTE: first tag is [1], not [0]
            pvno            [1] INTEGER (5) ,
            msg-type        [2] INTEGER (10 -- AS -- | 12 -- TGS --),
            padata          [3] SEQUENCE OF PA-DATA OPTIONAL
                                -- NOTE: not empty --,
            req-body        [4] KDC-REQ-BODY
        }

    Args:
        sequence: The ASN.1 sequence value as a dict to unpack.

    Attributes:
        padata (PAData): The pre-authentication data.
        req_body (KdcReqBody): The body of the request.

    .. _RFC 4120 5.4.1:
        https://www.rfc-editor.org/rfc/rfc4120#section-5.4.1
    """
    MESSAGE_TYPE = KerberosMessageType.as_req
    PARSE_MAP = [('pvno', 'PVNO', ParseType.default), ('msg-type', 'MESSAGE_TYPE', ParseType.enum), ('padata', 'padata', ParseType.token), ('req-body', 'req_body', ParseType.token)]

    def __init__(self, sequence: typing.Dict[int, ASN1Value]) -> None:

        def unpack_padata(value: typing.Union[ASN1Value, bytes]) -> typing.List:
            return [PAData.unpack(p) for p in unpack_asn1_sequence(value)]
        self.padata = get_sequence_value(sequence, 3, 'KDC-REQ', 'pa-data', unpack_padata)
        self.req_body = get_sequence_value(sequence, 4, 'KDC-REQ', 'req-body', KdcReqBody.unpack)