import dataclasses
import typing
from spnego._ntlm_raw.crypto import is_ntlm_hash
from spnego.exceptions import InvalidCredentialError, NoCredentialError
@dataclasses.dataclass
class KerberosKeytab:
    """Kerberos Keytab Credential.

    Used with :class:`GSSAPIProxy` for Kerberos authentication. It is used to
    retrieve a Kerberos ticket using a keytab for authentication rather than a
    password. The keytab value is specified in the form ``TYPE:RESIDUAL`` where
    the ``TYPE`` supported is down to the installed Kerberos/GSSAPI
    implementation and ``RESIDUAL`` is a value specific to the type. Common
    types are:

        FILE: The value is the path to a keytab.
        MEMORY: The value is a unique identifier to a keytab stored in memory
            of the current process. It must be resolvable by the linked GSSAPI
            provider that this library uses.

    There are other ccache types but they are mostly platform or GSSAPI
    implementation specific.

    .. Note:
        This only works on Linux, Windows does not have the concept of a
        keytab.

    Attributes:
        keytab: The keytab to use for authentication. The path will not be
            expanded of have variables substituted so should be the absolute
            path to the keytab.
        principal: The Kerberos principal to get the credential for. Should be
            in the UPN form `username@REALM.COM`. Set to `None` to use the
            first keytab entry.
    """
    keytab: str
    principal: typing.Optional[str] = None

    @property
    def supported_protocols(self) -> typing.List[str]:
        """List of protocols the credential can be used for."""
        return ['kerberos']