import binascii
import hashlib
import hmac
import struct
import ntlm_auth.compute_keys as compkeys
from ntlm_auth.constants import NegotiateFlags, SignSealConstants
from ntlm_auth.rc4 import ARC4
def _seal_message(self, message):
    """
        [MS-NLMP] v28.0 2016-07-14

        3.4.3 Message Confidentiality
        Will generate an encrypted message using RC4 based on the
        ClientSealingKey

        :param message: The message to be sealed (encrypted)
        :return encrypted_message: The encrypted message
        """
    encrypted_message = self.outgoing_handle.update(message)
    return encrypted_message