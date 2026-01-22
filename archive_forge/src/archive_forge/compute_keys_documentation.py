import hashlib
import hmac
from ntlm_auth.des import DES
from ntlm_auth.constants import NegotiateFlags

    3.4.5.3 SEALKEY
    Calculates the seal_key used to seal (encrypt) messages. This for
    authentication where NTLMSSP_NEGOTIATE_EXTENDED_SESSIONSECURITY has been
    negotiated. Will weaken the keys if NTLMSSP_NEGOTIATE_128 is not
    negotiated, will try NEGOTIATE_56 and then will default to the 40-bit key

    :param negotiate_flags: The negotiate_flags structure sent by the server
    :param exported_session_key: A 128-bit session key used to derive signing
        and sealing keys
    :param magic_constant: A constant value set in the MS-NLMP documentation
        (constants.SignSealConstants)
    :return seal_key: Key used to seal messages
    