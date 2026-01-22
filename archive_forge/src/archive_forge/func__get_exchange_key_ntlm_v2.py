import hashlib
import hmac
from ntlm_auth.des import DES
from ntlm_auth.constants import NegotiateFlags
def _get_exchange_key_ntlm_v2(session_base_key):
    """
    [MS-NLMP] v28.0 2016-07-14

    4.3.5.1 KXKEY
    Calculates the Key Exchange Key for NTLMv2 authentication. Used for signing
    and sealing messages. According to docs, 'If NTLM v2 is used,
    KeyExchangeKey MUST be set to the given 128-bit SessionBaseKey

    :param session_base_key: A session key calculated from the user password
        challenge
    :return key_exchange_key: The Key Exchange Key (KXKEY) used to sign and
        seal messages
    """
    return session_base_key