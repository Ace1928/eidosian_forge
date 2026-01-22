import enum
import struct
import typing
from spnego._asn1 import (
from spnego._context import GSSMech
from spnego._kerberos import KerberosV5Msg
from spnego._ntlm_raw.messages import NTLMMessage
class NegTokenInit:
    """The NegTokenInit GSSAPI value.

    This is the initial negotiation message token in a GSSAPI exchange. Typically the `NegTokenInit` value is sent
    when sending the first authentication token. The `NegTokenInit2` token is an extension that adds the `negHints`
    field. Unfortunately as the tag number for the `mechListMIC` is the same for `negHints` unpacking the value
    requires some extra checks.

    The ASN.1 definition for the NegTokenInit structure is defined in `RFC 4178 4.2.1`_::

        NegTokenInit ::= SEQUENCE {
            mechTypes       [0] MechTypeList,
            reqFlags        [1] ContextFlags  OPTIONAL,
            -- inherited from RFC 2478 for backward compatibility,
            -- RECOMMENDED to be left out
            mechToken       [2] OCTET STRING  OPTIONAL,
            mechListMIC     [3] OCTET STRING  OPTIONAL,
            ...
        }
        ContextFlags ::= BIT STRING {
            delegFlag       (0),
            mutualFlag      (1),
            replayFlag      (2),
            sequenceFlag    (3),
            anonFlag        (4),
            confFlag        (5),
            integFlag       (6)
        } (SIZE (32))

    The ASN.1 definition for the `NegTokenInit2`_ structure is defined as::

        NegHints ::= SEQUENCE {
            hintName[0] GeneralString OPTIONAL,
            hintAddress[1] OCTET STRING OPTIONAL
        }
        NegTokenInit2 ::= SEQUENCE {
            mechTypes[0] MechTypeList OPTIONAL,
            reqFlags [1] ContextFlags OPTIONAL,
            mechToken [2] OCTET STRING OPTIONAL,
            negHints [3] NegHints OPTIONAL,
            mechListMIC [4] OCTET STRING OPTIONAL,
            ...
        }

    Args:
        mech_types: One or more security mechanisms available for the initiator, in decreasing preference order.
        req_flags: Should be omitted, service options that are requested to establish the context.
        mech_token: Contains the optimistic mechanism token.
        hint_name: Used for the NegTokenInit2 structure only, should be omitted.
        hint_address: Used for the NegTokenINit2 structure only, should be omitted.
        mech_list_mic: The message integrity code (MIC) token.

    Attributes:
        mech_types (List[str]): See args.
        req_flags (ContextFlags): See args.
        mech_token (bytes): See args.
        hint_name (bytes): See args.
        hint_address (bytes): See args.
        mech_list_mic (bytes): See args.

    .. _RFC 4178 4.2.1:
        https://www.rfc-editor.org/rfc/rfc4178.html#section-4.2.1

    .. _NegTokenInit2:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-spng/8e71cf53-e867-4b79-b5b5-38c92be3d472
    """

    def __init__(self, mech_types: typing.Optional[typing.List[str]]=None, req_flags: typing.Optional[ContextFlags]=None, mech_token: typing.Optional[bytes]=None, hint_name: typing.Optional[bytes]=None, hint_address: typing.Optional[bytes]=None, mech_list_mic: typing.Optional[bytes]=None) -> None:
        self.mech_types = mech_types or []
        self.req_flags = req_flags
        self.mech_token = mech_token
        self.hint_name = hint_name
        self.hint_address = hint_address
        self.mech_list_mic = mech_list_mic

    def pack(self) -> bytes:
        """Packs the NegTokenInit as a byte string."""

        def pack_elements(value_map: typing.Iterable[typing.Tuple[int, typing.Any, typing.Callable]]) -> typing.List[bytes]:
            elements = []
            for tag, value, pack_func in value_map:
                if value is not None:
                    elements.append(pack_asn1(TagClass.context_specific, True, tag, pack_func(value)))
            return elements
        req_flags = struct.pack('B', self.req_flags) if self.req_flags is not None else None
        base_map: typing.List[typing.Tuple[int, typing.Any, typing.Callable]] = [(0, self.mech_types, pack_mech_type_list), (1, req_flags, pack_asn1_bit_string), (2, self.mech_token, pack_asn1_octet_string)]
        neg_hints = pack_elements([(0, self.hint_name, pack_asn1_general_string), (1, self.hint_address, pack_asn1_octet_string)])
        if neg_hints:
            base_map.append((3, neg_hints, pack_asn1_sequence))
            base_map.append((4, self.mech_list_mic, pack_asn1_octet_string))
        else:
            base_map.append((3, self.mech_list_mic, pack_asn1_octet_string))
        init_sequence = pack_elements(base_map)
        b_data = pack_asn1_sequence(init_sequence)
        return InitialContextToken(GSSMech.spnego.value, pack_asn1(TagClass.context_specific, True, 0, b_data)).pack()

    @staticmethod
    def unpack(b_data: bytes) -> 'NegTokenInit':
        """Unpacks the NegTokenInit TLV value."""
        neg_seq = unpack_asn1_tagged_sequence(unpack_asn1(b_data)[0])
        mech_types = [unpack_asn1_object_identifier(m) for m in get_sequence_value(neg_seq, 0, 'NegTokenInit', 'mechTypes', unpack_asn1_sequence) or []]
        req_flags = get_sequence_value(neg_seq, 1, 'NegTokenInit', 'reqFlags', unpack_asn1_bit_string)
        if req_flags:
            req_flags = ContextFlags(bytearray(req_flags)[-1])
        mech_token = get_sequence_value(neg_seq, 2, 'NegTokenInit', 'mechToken', unpack_asn1_octet_string)
        hint_name = hint_address = mech_list_mic = None
        if 3 in neg_seq:
            tag_class = neg_seq[3].tag_class
            tag_number = neg_seq[3].tag_number
            if tag_class == TagClass.universal and tag_number == TypeTagNumber.sequence:
                neg_hints = unpack_asn1_tagged_sequence(neg_seq[3].b_data)
                hint_name = get_sequence_value(neg_hints, 0, 'NegHints', 'hintName', unpack_asn1_general_string)
                hint_address = get_sequence_value(neg_hints, 1, 'NegHints', 'hintAddress', unpack_asn1_octet_string)
            else:
                mech_list_mic = get_sequence_value(neg_seq, 3, 'NegTokenInit', 'mechListMIC', unpack_asn1_octet_string)
        if not mech_list_mic:
            mech_list_mic = get_sequence_value(neg_seq, 4, 'NegTokenInit2', 'mechListMIC', unpack_asn1_octet_string)
        return NegTokenInit(mech_types, req_flags, mech_token, hint_name, hint_address, mech_list_mic)