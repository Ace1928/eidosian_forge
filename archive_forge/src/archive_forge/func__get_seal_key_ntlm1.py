import hashlib
import hmac
from ntlm_auth.des import DES
from ntlm_auth.constants import NegotiateFlags
def _get_seal_key_ntlm1(negotiate_flags, exported_session_key):
    """
    3.4.5.3 SEALKEY
    Calculates the seal_key used to seal (encrypt) messages. This for
    authentication where NTLMSSP_NEGOTIATE_EXTENDED_SESSIONSECURITY has not
    been negotiated. Will weaken the keys if NTLMSSP_NEGOTIATE_56 is not
    negotiated it will default to the 40-bit key

    :param negotiate_flags: The negotiate_flags structure sent by the server
    :param exported_session_key: A 128-bit session key used to derive signing
        and sealing keys
    :return seal_key: Key used to seal messages
    """
    if negotiate_flags & NegotiateFlags.NTLMSSP_NEGOTIATE_56:
        seal_key = exported_session_key[:7] + b'\xa0'
    else:
        seal_key = exported_session_key[:5] + b'\xe58\xb0'
    return seal_key