from Cryptodome.Signature.pss import MGF1
import Cryptodome.Hash.SHA1
from Cryptodome.Util.py3compat import _copy_bytes
import Cryptodome.Util.number
from Cryptodome.Util.number import ceil_div, bytes_to_long, long_to_bytes
from Cryptodome.Util.strxor import strxor
from Cryptodome import Random
from ._pkcs1_oaep_decode import oaep_decode
Decrypt a message with PKCS#1 OAEP.

        :param ciphertext: The encrypted message.
        :type ciphertext: bytes/bytearray/memoryview

        :returns: The original message (plaintext).
        :rtype: bytes

        :raises ValueError:
            if the ciphertext has the wrong length, or if decryption
            fails the integrity check (in which case, the decryption
            key is probably wrong).
        :raises TypeError:
            if the RSA key has no private half (i.e. you are trying
            to decrypt using a public key).
        