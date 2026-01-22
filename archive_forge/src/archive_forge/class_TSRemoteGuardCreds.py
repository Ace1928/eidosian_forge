import typing
from spnego._asn1 import (
class TSRemoteGuardCreds:
    """CredSSP TSRemoteGuardCreds structure.

    The TSRemoteGuardCreds structure contains a logon credential and supplemental credentials provided by security
    packages. The format of the individual credentials depends on the packages that provided them.

    The ASN.1 definition for the TSRemoteGuardCreds structure is defined in `MS-CSSP 2.2.1.2.3 TSRemoteGuardCreds`_::

        TSRemoteGuardCreds ::= SEQUENCE{
            logonCred           [0] TSRemoteGuardPackageCred,
            supplementalCreds   [1] SEQUENCE OF TSRemoteGuardPackageCred OPTIONAL,
        }

    Args:
        logon_cred: The logon credential for the user.
        supplemental_creds: Optional supplemental credentials for other security packages.

    Attributes:
        logon_cred (TSRemoteGuardPackageCred): See args.
        supplemental_creds (List[TSRemoteGuardPackageCred]): See args.

    .. _MS-CSSP 2.2.1.2.3 TSRemoteGuardCreds:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-cssp/7ef8229c-44ea-4c1b-867f-00369b882b38
    """

    def __init__(self, logon_cred: 'TSRemoteGuardPackageCred', supplemental_creds: typing.Optional[typing.Union['TSRemoteGuardPackageCred', typing.List['TSRemoteGuardPackageCred']]]=None) -> None:
        self.logon_cred = logon_cred
        if supplemental_creds is not None and (not isinstance(supplemental_creds, list)):
            supplemental_creds = [supplemental_creds]
        self.supplemental_creds = supplemental_creds

    def pack(self) -> bytes:
        """Packs the TSRemoteGuardCreds as a byte string."""
        elements = [pack_asn1(TagClass.context_specific, True, 0, self.logon_cred.pack())]
        if self.supplemental_creds is not None:
            supplemental_creds = [cred.pack() for cred in self.supplemental_creds]
            elements.append(pack_asn1(TagClass.context_specific, True, 1, pack_asn1_sequence(supplemental_creds)))
        return pack_asn1_sequence(elements)

    @staticmethod
    def unpack(b_data: typing.Union[ASN1Value, bytes]) -> 'TSRemoteGuardCreds':
        """Unpacks the TSRemoteGuardCreds TLV value."""
        cred = unpack_sequence(b_data)
        logon_cred = get_sequence_value(cred, 0, 'TSRemoteGuardCreds', 'logonCred', TSRemoteGuardPackageCred.unpack)
        raw_supplemental_creds = get_sequence_value(cred, 1, 'TSRemoteGuardCreds', 'supplementalCreds')
        if raw_supplemental_creds:
            supplemental_creds = []
            remaining_bytes = raw_supplemental_creds.b_data
            while remaining_bytes:
                supplemental_creds.append(TSRemoteGuardPackageCred.unpack(remaining_bytes))
                remaining_bytes = unpack_asn1(remaining_bytes)[1]
        else:
            supplemental_creds = None
        return TSRemoteGuardCreds(logon_cred, supplemental_creds)