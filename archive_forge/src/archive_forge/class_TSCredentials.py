import typing
from spnego._asn1 import (
class TSCredentials:
    """CredSSP TSCredentials structure.

    The TSCredentials structure contains both the user's credentials that are delegated to the server and their type.

    The ASN.1 definition for the TSCredentials structure is defined in `MS-CSSP 2.2.1.2 TSCredentials`_::

        TSCredentials ::= SEQUENCE {
                credType    [0] INTEGER,
                credentials [1] OCTET STRING
        }

    Args:
        credentials: The credential structure; TSPasswordCreds, TSSmartCardCreds, TSRemoteGuardCreds.

    Attributes:
        credentials (Union[TSPasswordCreds, TSSmartCardCreds, TSRemoteGuardCreds]): See args.

    .. _MS-CSSP 2.2.1.2 TSCredentials:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-cssp/94a1ab00-5500-42fd-8d3d-7a84e6c2cf03
    """

    def __init__(self, credentials: typing.Union['TSPasswordCreds', 'TSSmartCardCreds', 'TSRemoteGuardCreds']) -> None:
        self.credentials = credentials

    @property
    def cred_type(self) -> int:
        """The credential type of credentials as an integer."""
        if isinstance(self.credentials, TSPasswordCreds):
            return 1
        elif isinstance(self.credentials, TSSmartCardCreds):
            return 2
        elif isinstance(self.credentials, TSRemoteGuardCreds):
            return 6
        else:
            raise ValueError('Invalid credential type set')

    def pack(self) -> bytes:
        """Packs the TSCredentials as a byte string."""
        cred_type = self.cred_type
        credentials = self.credentials.pack()
        return pack_asn1_sequence([pack_asn1(TagClass.context_specific, True, 0, pack_asn1_integer(cred_type)), pack_asn1(TagClass.context_specific, True, 1, pack_asn1_octet_string(credentials))])

    @staticmethod
    def unpack(b_data: typing.Union[ASN1Value, bytes]) -> 'TSCredentials':
        """Unpacks the TSCredentials TLV value."""
        credential = unpack_sequence(b_data)
        cred_type = get_sequence_value(credential, 0, 'TSCredentials', 'credType', unpack_asn1_integer)
        credentials_raw = get_sequence_value(credential, 1, 'TSCredentials', 'credentials', unpack_asn1_octet_string)
        cred_class: typing.Optional[typing.Union[typing.Type[TSPasswordCreds], typing.Type[TSSmartCardCreds], typing.Type[TSRemoteGuardCreds]]] = {1: TSPasswordCreds, 2: TSSmartCardCreds, 6: TSRemoteGuardCreds}.get(cred_type)
        if not cred_class:
            raise ValueError('Unknown credType %s in TSCredentials, cannot unpack' % cred_type)
        credentials = cred_class.unpack(credentials_raw)
        return TSCredentials(credentials)