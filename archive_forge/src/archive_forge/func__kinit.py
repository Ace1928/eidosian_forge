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
def _kinit(username: bytes, password: bytes, forwardable: typing.Optional[bool]=None, is_keytab: bool=False) -> 'gssapi.raw.Creds':
    """Gets a Kerberos credential.

    This will get the GSSAPI credential that contains the Kerberos TGT inside
    it. This is used instead of gss_acquire_cred_with_password as the latter
    does not expose a way to request a forwardable ticket or to retrieve a TGT
    from a keytab. This way makes it possible to request whatever is needed
    before making it usable in GSSAPI.

    Args:
        username: The username to get the credential for.
        password: The password to use to retrieve the credential.
        forwardable: Whether to request a forwardable credential.
        is_keytab: Whether password is a keytab or just a password.

    Returns:
        gssapi.raw.Creds: The GSSAPI credential for the Kerberos mech.
    """
    ctx = krb5.init_context()
    kt: typing.Optional[krb5.KeyTab] = None
    princ: typing.Optional[krb5.Principal] = None
    if is_keytab:
        kt = krb5.kt_resolve(ctx, password)
        if not username:
            first_entry = list(kt)[0]
            princ = copy.copy(first_entry.principal)
    if not princ:
        princ = krb5.parse_name_flags(ctx, username)
    init_opt = krb5.get_init_creds_opt_alloc(ctx)
    if hasattr(krb5, 'get_init_creds_opt_set_default_flags'):
        realm = krb5.principal_get_realm(ctx, princ)
        krb5.get_init_creds_opt_set_default_flags(ctx, init_opt, b'gss_krb5', realm)
    krb5.get_init_creds_opt_set_canonicalize(init_opt, True)
    if forwardable is not None:
        krb5.get_init_creds_opt_set_forwardable(init_opt, forwardable)
    if kt:
        cred = krb5.get_init_creds_keytab(ctx, princ, init_opt, keytab=kt)
    else:
        cred = krb5.get_init_creds_password(ctx, princ, init_opt, password=password)
    mem_ccache = krb5.cc_new_unique(ctx, b'MEMORY')
    krb5.cc_initialize(ctx, mem_ccache, princ)
    krb5.cc_store_cred(ctx, mem_ccache, cred)
    return _gss_acquire_cred_from_ccache(mem_ccache, None)