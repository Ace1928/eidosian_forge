import base64
import copy
import logging
import sys
import typing
from spnego._context import (
from spnego._credential import (
from spnego._text import to_bytes, to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import GSSError as NativeError
from spnego.exceptions import (
from spnego.iov import BufferType, IOVBuffer, IOVResBuffer
def _gss_acquire_cred_from_ccache(ccache: 'krb5.CCache', principal: typing.Optional['krb5.Principal']) -> 'gssapi.raw.Creds':
    """Acquire GSSAPI credential from CCache.

    Args:
        ccache: The CCache to acquire the credential from.
        principal: The optional principal to acquire the cred for.

    Returns:
        gssapi.raw.Creds: The GSSAPI credentials from the ccache.
    """
    if hasattr(gssapi.raw, 'acquire_cred_from'):
        kerberos = gssapi.OID.from_int_seq(GSSMech.kerberos.value)
        name = None
        if principal:
            name = gssapi.Name(base=to_text(principal.name), name_type=gssapi.NameType.user)
        ccache_name = ccache.name or b''
        if ccache.cache_type:
            ccache_name = ccache.cache_type + b':' + ccache_name
        return gssapi.raw.acquire_cred_from({b'ccache': ccache_name}, name=name, mechs=[kerberos], usage='initiate').creds
    else:
        gssapi_creds = gssapi.raw.Creds()
        gssapi.raw.krb5_import_cred(gssapi_creds, cache=ccache.addr, keytab_principal=principal.addr if principal else None)
        return gssapi_creds