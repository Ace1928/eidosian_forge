import typing
from spnego._asn1 import (
class TSSmartCardCreds:
    """CredSSP TSSmartCardCreds structure.

    The TSSmartCardCreds structure contains the user's smart card credentials that are delegated to the server.

    The ASN.1 definition for the TSSmartCardCreds structure is defined in `MS-CSSP 2.2.1.2.2 TSSmartCardCreds`_::

        TSSmartCardCreds ::= SEQUENCE {
                pin         [0] OCTET STRING,
                cspData     [1] TSCspDataDetail,
                userHint    [2] OCTET STRING OPTIONAL,
                domainHint  [3] OCTET STRING OPTIONAL
        }

    Args:
        pin: THe user's smart card PIN.
        csp_data: Info about the cryptographic service provider (CSP).
        user_hint: The user's account hint.
        domain_hint: The user's domain name to which the user's account belongs.

    Attributes:
        pin (str): See args.
        csp_data (TSCspDataDetail): See args.
        user_hint (Optional[str]): See args.
        domain_hint (Optional[str]): See args.

    .. _MS-CSSP 2.2.1.2.2 TSSmartCardCreds:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-cssp/4251d165-cf01-4513-a5d8-39ee4a98b7a4
    """

    def __init__(self, pin: str, csp_data: 'TSCspDataDetail', user_hint: typing.Optional[str]=None, domain_hint: typing.Optional[str]=None) -> None:
        self.pin = pin
        self.csp_data = csp_data
        self.user_hint = user_hint
        self.domain_hint = domain_hint

    def pack(self) -> bytes:
        """Packs the TSSmartCardCreds as a byte string."""
        elements = [pack_asn1(TagClass.context_specific, True, 0, pack_asn1_octet_string(self.pin.encode('utf-16-le'))), pack_asn1(TagClass.context_specific, True, 1, self.csp_data.pack())]
        for idx, value in [(2, self.user_hint), (3, self.domain_hint)]:
            if value:
                b_value = value.encode('utf-16-le')
                elements.append(pack_asn1(TagClass.context_specific, True, idx, pack_asn1_octet_string(b_value)))
        return pack_asn1_sequence(elements)

    @staticmethod
    def unpack(b_data: typing.Union[ASN1Value, bytes]) -> 'TSSmartCardCreds':
        """Unpacks the TSSmartCardCreds TLV value."""
        creds = unpack_sequence(b_data)
        pin = unpack_text_field(creds, 0, 'TSSmartCardCreds', 'pin') or ''
        csp_data = get_sequence_value(creds, 1, 'TSSmartCardCreds', 'cspData', TSCspDataDetail.unpack)
        user_hint = unpack_text_field(creds, 2, 'TSSmartCardCreds', 'userHint', default=None)
        domain_hint = unpack_text_field(creds, 3, 'TSSmartCardCreds', 'domainHint', default=None)
        return TSSmartCardCreds(pin, csp_data, user_hint, domain_hint)