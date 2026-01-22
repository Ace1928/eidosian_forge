import enum
import struct
import typing
from spnego._asn1 import (
from spnego._context import GSSMech
from spnego._kerberos import KerberosV5Msg
from spnego._ntlm_raw.messages import NTLMMessage
class NegTokenResp:
    """The NegTokenResp GSSAPI value.

    This is the message token in a GSSAPI exchange that is used for subsequent messages after the `NegTokenInit` has
    been exchanged.

    The ASN.1 definition for the NegTokenResp structure is defined in `RFC 4178 4.2.2`_::

        NegTokenResp ::= SEQUENCE {
            negState       [0] ENUMERATED {
                accept-completed    (0),
                accept-incomplete   (1),
                reject              (2),
                request-mic         (3)
            }                                 OPTIONAL,
            -- REQUIRED in the first reply from the target
            supportedMech   [1] MechType      OPTIONAL,
            -- present only in the first reply from the target
            responseToken   [2] OCTET STRING  OPTIONAL,
            mechListMIC     [3] OCTET STRING  OPTIONAL,
            ...
        }

    Args:
        neg_state: The state of the negotiation.
        supported_mech: Should only be present in the first reply, must be one of the mech(s) offered by the initiator.
        response_token: Contains the token specific to the mechanism selected.
        mech_list_mic: The message integrity code (MIC) token.

    Attributes:
        neg_state (NegState): See args.
        supported_mech (str): See args.
        response_token (bytes): See args.
        mech_list_mic (bytes): See args.

    .. _RFC 4178 4.2.2:
        https://www.rfc-editor.org/rfc/rfc4178.html#section-4.2.2
    """

    def __init__(self, neg_state: typing.Optional[NegState]=None, supported_mech: typing.Optional[str]=None, response_token: typing.Optional[bytes]=None, mech_list_mic: typing.Optional[bytes]=None) -> None:
        self.neg_state = neg_state
        self.supported_mech = supported_mech
        self.response_token = response_token
        self.mech_list_mic = mech_list_mic

    def pack(self) -> bytes:
        """Packs the NegTokenResp as a byte string."""
        value_map: typing.List[typing.Tuple[int, typing.Any, typing.Callable[[typing.Any], bytes]]] = [(0, self.neg_state, pack_asn1_enumerated), (1, self.supported_mech, pack_asn1_object_identifier), (2, self.response_token, pack_asn1_octet_string), (3, self.mech_list_mic, pack_asn1_octet_string)]
        elements = []
        for tag, value, pack_func in value_map:
            if value is not None:
                elements.append(pack_asn1(TagClass.context_specific, True, tag, pack_func(value)))
        b_data = pack_asn1_sequence(elements)
        return pack_asn1(TagClass.context_specific, True, 1, b_data)

    @staticmethod
    def unpack(b_data: bytes) -> 'NegTokenResp':
        """Unpacks the NegTokenResp TLV value."""
        neg_seq = unpack_asn1_tagged_sequence(unpack_asn1(b_data)[0])
        neg_state = get_sequence_value(neg_seq, 0, 'NegTokenResp', 'negState', unpack_asn1_enumerated)
        if neg_state is not None:
            neg_state = NegState(neg_state)
        supported_mech = get_sequence_value(neg_seq, 1, 'NegTokenResp', 'supportedMech', unpack_asn1_object_identifier)
        response_token = get_sequence_value(neg_seq, 2, 'NegTokenResp', 'responseToken', unpack_asn1_octet_string)
        mech_list_mic = get_sequence_value(neg_seq, 3, 'NegTokenResp', 'mechListMIC', unpack_asn1_octet_string)
        return NegTokenResp(neg_state, supported_mech, response_token, mech_list_mic)