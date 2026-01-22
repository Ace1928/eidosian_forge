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
def _gss_sasl_description(mech: 'gssapi.OID') -> typing.Optional[bytes]:
    """Attempts to get the SASL description of the mech specified."""
    try:
        res = _gss_sasl_description.result
        return res[mech.dotted_form]
    except (AttributeError, KeyError):
        res = getattr(_gss_sasl_description, 'result', {})
    try:
        sasl_desc = gssapi.raw.inquire_saslname_for_mech(mech).mech_description
    except Exception as e:
        log.debug('gss_inquire_saslname_for_mech(%s) failed: %s' % (mech.dotted_form, str(e)))
        sasl_desc = None
    res[mech.dotted_form] = sasl_desc
    _gss_sasl_description.result = res
    return _gss_sasl_description(mech)