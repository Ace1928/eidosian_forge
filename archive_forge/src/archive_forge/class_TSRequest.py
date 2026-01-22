import typing
from spnego._asn1 import (
class TSRequest:
    """CredSSP TSRequest structure.

    The TSRequest structure is the top-most structure used by the CredSSP client and CredSSP server. The TSRequest
    message is always sent over the TLS-encrypted channel between the client and server in a CredSSP protocol exchange.

    The ASN.1 definition for the TSRequest structure is defined in `MS-CSSP 2.2.1 TSRequest`_::

        TSRequest ::= SEQUENCE {
                version    [0] INTEGER,
                negoTokens [1] NegoData  OPTIONAL,
                authInfo   [2] OCTET STRING OPTIONAL,
                pubKeyAuth [3] OCTET STRING OPTIONAL,
                errorCode  [4] INTEGER OPTIONAL,
                clientNonce [5] OCTET STRING OPTIONAL
        }

    Args:
        version: THe supported version of the CredSSP protocol.
        nego_tokens: A list of NegoData structures that contains te SPNEGO tokens.
        auth_info: The encrypted TSCredentials structure that contains the user's credentials to be delegated to the
            server.
        pub_key_auth: The encrypted public key used in the TLS handshake between the client and the server.
        error_code: The error code that represents the NTSTATUS failure code from the server.
        client_nonce: A 32-byte array of cryptograpically random bytes for the pub_key_auth hash computation (version
            or above).

    Attributes:
        version (int): See args.
        nego_tokens (Optional[List[int]]): See args.
        auth_info (bytes): See args.
        pub_key_auth (bytes): See args.
        error_code (int): See args.
        client_nonce (bytes): See args.

    .. _MS-CSSP 2.2.1 TSRequest:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-cssp/9664994d-0784-4659-b85b-83b8d54c2336
    """

    def __init__(self, version: int, nego_tokens: typing.Optional[typing.Union[NegoData, typing.List[NegoData]]]=None, auth_info: typing.Optional[bytes]=None, pub_key_auth: typing.Optional[bytes]=None, error_code: typing.Optional[int]=None, client_nonce: typing.Optional[bytes]=None) -> None:
        self.version = version
        if nego_tokens is not None and (not isinstance(nego_tokens, list)):
            nego_tokens = [nego_tokens]
        self.nego_tokens = nego_tokens
        self.auth_info = auth_info
        self.pub_key_auth = pub_key_auth
        self.error_code = error_code
        self.client_nonce = client_nonce

    def pack(self) -> bytes:
        """Packs the TSRequest as a byte string."""
        elements = [pack_asn1(TagClass.context_specific, True, 0, pack_asn1_integer(self.version))]
        if self.nego_tokens:
            nego_tokens = [token.pack() for token in self.nego_tokens]
            elements.append(pack_asn1(TagClass.context_specific, True, 1, pack_asn1_sequence(nego_tokens)))
        value_map: typing.List[typing.Tuple[int, typing.Any, typing.Callable]] = [(2, self.auth_info, pack_asn1_octet_string), (3, self.pub_key_auth, pack_asn1_octet_string), (4, self.error_code, pack_asn1_integer), (5, self.client_nonce, pack_asn1_octet_string)]
        for tag, value, pack_func in value_map:
            if value is not None:
                elements.append(pack_asn1(TagClass.context_specific, True, tag, pack_func(value)))
        return pack_asn1_sequence(elements)

    @staticmethod
    def unpack(b_data: typing.Union[ASN1Value, bytes]) -> 'TSRequest':
        """Unpacks the TSRequest TLV value."""
        request = unpack_sequence(b_data)
        version = get_sequence_value(request, 0, 'TSRequest', 'version', unpack_asn1_integer)
        nego_tokens = get_sequence_value(request, 1, 'TSRequest', 'negoTokens')
        if nego_tokens is not None:
            remaining_bytes = nego_tokens.b_data
            nego_tokens = []
            while remaining_bytes:
                nego_tokens.append(NegoData.unpack(remaining_bytes))
                remaining_bytes = unpack_asn1(remaining_bytes)[1]
        auth_info = get_sequence_value(request, 2, 'TSRequest', 'authInfo', unpack_asn1_octet_string)
        pub_key_auth = get_sequence_value(request, 3, 'TSRequest', 'pubKeyAuth', unpack_asn1_octet_string)
        error_code = get_sequence_value(request, 4, 'TSRequest', 'errorCode', unpack_asn1_integer)
        client_nonce = get_sequence_value(request, 5, 'TSRequest', 'clientNonce', unpack_asn1_octet_string)
        return TSRequest(version, nego_tokens=nego_tokens, auth_info=auth_info, pub_key_auth=pub_key_auth, error_code=error_code, client_nonce=client_nonce)