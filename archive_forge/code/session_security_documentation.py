import binascii
import hashlib
import hmac
import struct
import ntlm_auth.compute_keys as compkeys
from ntlm_auth.constants import NegotiateFlags, SignSealConstants
from ntlm_auth.rc4 import ARC4

        Will verify that the signature received from the server matches up with
        the expected signature computed locally. Will throw an exception if
        they do not match

        :param message: The message data that is received from the server
        :param signature: The signature of the message received from the server
        