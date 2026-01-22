import binascii
import struct
from Cryptodome import Random
from Cryptodome.Util.py3compat import tobytes, bord, tostr
from Cryptodome.Util.asn1 import DerSequence, DerNull
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math.Primality import (test_probable_prime,
from Cryptodome.PublicKey import (_expand_subject_public_key_info,
importKey = import_key
class RsaKey(object):
    """Class defining an RSA key, private or public.
    Do not instantiate directly.
    Use :func:`generate`, :func:`construct` or :func:`import_key` instead.

    :ivar n: RSA modulus
    :vartype n: integer

    :ivar e: RSA public exponent
    :vartype e: integer

    :ivar d: RSA private exponent
    :vartype d: integer

    :ivar p: First factor of the RSA modulus
    :vartype p: integer

    :ivar q: Second factor of the RSA modulus
    :vartype q: integer

    :ivar invp: Chinese remainder component (:math:`p^{-1} \\text{mod } q`)
    :vartype invp: integer

    :ivar invq: Chinese remainder component (:math:`q^{-1} \\text{mod } p`)
    :vartype invq: integer

    :ivar u: Same as ``invp``
    :vartype u: integer
    """

    def __init__(self, **kwargs):
        """Build an RSA key.

        :Keywords:
          n : integer
            The modulus.
          e : integer
            The public exponent.
          d : integer
            The private exponent. Only required for private keys.
          p : integer
            The first factor of the modulus. Only required for private keys.
          q : integer
            The second factor of the modulus. Only required for private keys.
          u : integer
            The CRT coefficient (inverse of p modulo q). Only required for
            private keys.
        """
        input_set = set(kwargs.keys())
        public_set = set(('n', 'e'))
        private_set = public_set | set(('p', 'q', 'd', 'u'))
        if input_set not in (private_set, public_set):
            raise ValueError('Some RSA components are missing')
        for component, value in kwargs.items():
            setattr(self, '_' + component, value)
        if input_set == private_set:
            self._dp = self._d % (self._p - 1)
            self._dq = self._d % (self._q - 1)
            self._invq = None

    @property
    def n(self):
        return int(self._n)

    @property
    def e(self):
        return int(self._e)

    @property
    def d(self):
        if not self.has_private():
            raise AttributeError('No private exponent available for public keys')
        return int(self._d)

    @property
    def p(self):
        if not self.has_private():
            raise AttributeError("No CRT component 'p' available for public keys")
        return int(self._p)

    @property
    def q(self):
        if not self.has_private():
            raise AttributeError("No CRT component 'q' available for public keys")
        return int(self._q)

    @property
    def dp(self):
        if not self.has_private():
            raise AttributeError("No CRT component 'dp' available for public keys")
        return int(self._dp)

    @property
    def dq(self):
        if not self.has_private():
            raise AttributeError("No CRT component 'dq' available for public keys")
        return int(self._dq)

    @property
    def invq(self):
        if not self.has_private():
            raise AttributeError("No CRT component 'invq' available for public keys")
        if self._invq is None:
            self._invq = self._q.inverse(self._p)
        return int(self._invq)

    @property
    def invp(self):
        return self.u

    @property
    def u(self):
        if not self.has_private():
            raise AttributeError("No CRT component 'u' available for public keys")
        return int(self._u)

    def size_in_bits(self):
        """Size of the RSA modulus in bits"""
        return self._n.size_in_bits()

    def size_in_bytes(self):
        """The minimal amount of bytes that can hold the RSA modulus"""
        return (self._n.size_in_bits() - 1) // 8 + 1

    def _encrypt(self, plaintext):
        if not 0 <= plaintext < self._n:
            raise ValueError('Plaintext too large')
        return int(pow(Integer(plaintext), self._e, self._n))

    def _decrypt_to_bytes(self, ciphertext):
        if not 0 <= ciphertext < self._n:
            raise ValueError('Ciphertext too large')
        if not self.has_private():
            raise TypeError('This is not a private key')
        r = Integer.random_range(min_inclusive=1, max_exclusive=self._n)
        cp = Integer(ciphertext) * pow(r, self._e, self._n) % self._n
        m1 = pow(cp, self._dp, self._p)
        m2 = pow(cp, self._dq, self._q)
        h = (m2 - m1) * self._u % self._q
        mp = h * self._p + m1
        result = Integer._mult_modulo_bytes(r.inverse(self._n), mp, self._n)
        return result

    def _decrypt(self, ciphertext):
        """Legacy private method"""
        return bytes_to_long(self._decrypt_to_bytes(ciphertext))

    def has_private(self):
        """Whether this is an RSA private key"""
        return hasattr(self, '_d')

    def can_encrypt(self):
        return True

    def can_sign(self):
        return True

    def public_key(self):
        """A matching RSA public key.

        Returns:
            a new :class:`RsaKey` object
        """
        return RsaKey(n=self._n, e=self._e)

    def __eq__(self, other):
        if self.has_private() != other.has_private():
            return False
        if self.n != other.n or self.e != other.e:
            return False
        if not self.has_private():
            return True
        return self.d == other.d

    def __ne__(self, other):
        return not self == other

    def __getstate__(self):
        from pickle import PicklingError
        raise PicklingError

    def __repr__(self):
        if self.has_private():
            extra = ', d=%d, p=%d, q=%d, u=%d' % (int(self._d), int(self._p), int(self._q), int(self._u))
        else:
            extra = ''
        return 'RsaKey(n=%d, e=%d%s)' % (int(self._n), int(self._e), extra)

    def __str__(self):
        if self.has_private():
            key_type = 'Private'
        else:
            key_type = 'Public'
        return '%s RSA key at 0x%X' % (key_type, id(self))

    def export_key(self, format='PEM', passphrase=None, pkcs=1, protection=None, randfunc=None, prot_params=None):
        """Export this RSA key.

        Keyword Args:
          format (string):
            The desired output format:

            - ``'PEM'``. (default) Text output, according to `RFC1421`_/`RFC1423`_.
            - ``'DER'``. Binary output.
            - ``'OpenSSH'``. Text output, according to the OpenSSH specification.
              Only suitable for public keys (not private keys).

            Note that PEM contains a DER structure.

          passphrase (bytes or string):
            (*Private keys only*) The passphrase to protect the
            private key.

          pkcs (integer):
            (*Private keys only*) The standard to use for
            serializing the key: PKCS#1 or PKCS#8.

            With ``pkcs=1`` (*default*), the private key is encoded with a
            simple `PKCS#1`_ structure (``RSAPrivateKey``). The key cannot be
            securely encrypted.

            With ``pkcs=8``, the private key is encoded with a `PKCS#8`_ structure
            (``PrivateKeyInfo``). PKCS#8 offers the best ways to securely
            encrypt the key.

            .. note::
                This parameter is ignored for a public key.
                For DER and PEM, the output is always an
                ASN.1 DER ``SubjectPublicKeyInfo`` structure.

          protection (string):
            (*For private keys only*)
            The encryption scheme to use for protecting the private key
            using the passphrase.

            You can only specify a value if ``pkcs=8``.
            For all possible protection schemes,
            refer to :ref:`the encryption parameters of PKCS#8<enc_params>`.
            The recommended value is
            ``'PBKDF2WithHMAC-SHA512AndAES256-CBC'``.

            If ``None`` (default), the behavior depends on :attr:`format`:

            - if ``format='PEM'``, the obsolete PEM encryption scheme is used.
              It is based on MD5 for key derivation, and 3DES for encryption.

            - if ``format='DER'``, the ``'PBKDF2WithHMAC-SHA1AndDES-EDE3-CBC'``
              scheme is used.

          prot_params (dict):
            (*For private keys only*)

            The parameters to use to derive the encryption key
            from the passphrase. ``'protection'`` must be also specified.
            For all possible values,
            refer to :ref:`the encryption parameters of PKCS#8<enc_params>`.
            The recommendation is to use ``{'iteration_count':21000}`` for PBKDF2,
            and ``{'iteration_count':131072}`` for scrypt.

          randfunc (callable):
            A function that provides random bytes. Only used for PEM encoding.
            The default is :func:`Cryptodome.Random.get_random_bytes`.

        Returns:
          bytes: the encoded key

        Raises:
          ValueError:when the format is unknown or when you try to encrypt a private
            key with *DER* format and PKCS#1.

        .. warning::
            If you don't provide a pass phrase, the private key will be
            exported in the clear!

        .. _RFC1421:    http://www.ietf.org/rfc/rfc1421.txt
        .. _RFC1423:    http://www.ietf.org/rfc/rfc1423.txt
        .. _`PKCS#1`:   http://www.ietf.org/rfc/rfc3447.txt
        .. _`PKCS#8`:   http://www.ietf.org/rfc/rfc5208.txt
        """
        if passphrase is not None:
            passphrase = tobytes(passphrase)
        if randfunc is None:
            randfunc = Random.get_random_bytes
        if format == 'OpenSSH':
            e_bytes, n_bytes = [x.to_bytes() for x in (self._e, self._n)]
            if bord(e_bytes[0]) & 128:
                e_bytes = b'\x00' + e_bytes
            if bord(n_bytes[0]) & 128:
                n_bytes = b'\x00' + n_bytes
            keyparts = [b'ssh-rsa', e_bytes, n_bytes]
            keystring = b''.join([struct.pack('>I', len(kp)) + kp for kp in keyparts])
            return b'ssh-rsa ' + binascii.b2a_base64(keystring)[:-1]
        if self.has_private():
            binary_key = DerSequence([0, self.n, self.e, self.d, self.p, self.q, self.d % (self.p - 1), self.d % (self.q - 1), Integer(self.q).inverse(self.p)]).encode()
            if pkcs == 1:
                key_type = 'RSA PRIVATE KEY'
                if format == 'DER' and passphrase:
                    raise ValueError('PKCS#1 private key cannot be encrypted')
            else:
                from Cryptodome.IO import PKCS8
                if format == 'PEM' and protection is None:
                    key_type = 'PRIVATE KEY'
                    binary_key = PKCS8.wrap(binary_key, oid, None, key_params=DerNull())
                else:
                    key_type = 'ENCRYPTED PRIVATE KEY'
                    if not protection:
                        if prot_params:
                            raise ValueError("'protection' parameter must be set")
                        protection = 'PBKDF2WithHMAC-SHA1AndDES-EDE3-CBC'
                    binary_key = PKCS8.wrap(binary_key, oid, passphrase, protection, prot_params=prot_params, key_params=DerNull())
                    passphrase = None
        else:
            key_type = 'PUBLIC KEY'
            binary_key = _create_subject_public_key_info(oid, DerSequence([self.n, self.e]), DerNull())
        if format == 'DER':
            return binary_key
        if format == 'PEM':
            from Cryptodome.IO import PEM
            pem_str = PEM.encode(binary_key, key_type, passphrase, randfunc)
            return tobytes(pem_str)
        raise ValueError("Unknown key format '%s'. Cannot export the RSA key." % format)

    def exportKey(self, *args, **kwargs):
        """:meta private:"""
        return self.export_key(*args, **kwargs)

    def publickey(self):
        """:meta private:"""
        return self.public_key()

    def sign(self, M, K):
        """:meta private:"""
        raise NotImplementedError('Use module Cryptodome.Signature.pkcs1_15 instead')

    def verify(self, M, signature):
        """:meta private:"""
        raise NotImplementedError('Use module Cryptodome.Signature.pkcs1_15 instead')

    def encrypt(self, plaintext, K):
        """:meta private:"""
        raise NotImplementedError('Use module Cryptodome.Cipher.PKCS1_OAEP instead')

    def decrypt(self, ciphertext):
        """:meta private:"""
        raise NotImplementedError('Use module Cryptodome.Cipher.PKCS1_OAEP instead')

    def blind(self, M, B):
        """:meta private:"""
        raise NotImplementedError

    def unblind(self, M, B):
        """:meta private:"""
        raise NotImplementedError

    def size(self):
        """:meta private:"""
        raise NotImplementedError