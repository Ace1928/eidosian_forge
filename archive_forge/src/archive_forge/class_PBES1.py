import re
from Cryptodome import Hash
from Cryptodome import Random
from Cryptodome.Util.asn1 import (
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad
from Cryptodome.Protocol.KDF import PBKDF1, PBKDF2, scrypt
class PBES1(object):
    """Deprecated encryption scheme with password-based key derivation
    (originally defined in PKCS#5 v1.5, but still present in `v2.0`__).

    .. __: http://www.ietf.org/rfc/rfc2898.txt
    """

    @staticmethod
    def decrypt(data, passphrase):
        """Decrypt a piece of data using a passphrase and *PBES1*.

        The algorithm to use is automatically detected.

        :Parameters:
          data : byte string
            The piece of data to decrypt.
          passphrase : byte string
            The passphrase to use for decrypting the data.
        :Returns:
          The decrypted data, as a binary string.
        """
        enc_private_key_info = DerSequence().decode(data)
        encrypted_algorithm = DerSequence().decode(enc_private_key_info[0])
        encrypted_data = DerOctetString().decode(enc_private_key_info[1]).payload
        pbe_oid = DerObjectId().decode(encrypted_algorithm[0]).value
        cipher_params = {}
        if pbe_oid == _OID_PBE_WITH_MD5_AND_DES_CBC:
            from Cryptodome.Hash import MD5
            from Cryptodome.Cipher import DES
            hashmod = MD5
            module = DES
        elif pbe_oid == _OID_PBE_WITH_MD5_AND_RC2_CBC:
            from Cryptodome.Hash import MD5
            from Cryptodome.Cipher import ARC2
            hashmod = MD5
            module = ARC2
            cipher_params['effective_keylen'] = 64
        elif pbe_oid == _OID_PBE_WITH_SHA1_AND_DES_CBC:
            from Cryptodome.Hash import SHA1
            from Cryptodome.Cipher import DES
            hashmod = SHA1
            module = DES
        elif pbe_oid == _OID_PBE_WITH_SHA1_AND_RC2_CBC:
            from Cryptodome.Hash import SHA1
            from Cryptodome.Cipher import ARC2
            hashmod = SHA1
            module = ARC2
            cipher_params['effective_keylen'] = 64
        else:
            raise PbesError('Unknown OID for PBES1')
        pbe_params = DerSequence().decode(encrypted_algorithm[1], nr_elements=2)
        salt = DerOctetString().decode(pbe_params[0]).payload
        iterations = pbe_params[1]
        key_iv = PBKDF1(passphrase, salt, 16, iterations, hashmod)
        key, iv = (key_iv[:8], key_iv[8:])
        cipher = module.new(key, module.MODE_CBC, iv, **cipher_params)
        pt = cipher.decrypt(encrypted_data)
        return unpad(pt, cipher.block_size)