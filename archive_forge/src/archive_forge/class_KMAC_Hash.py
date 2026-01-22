from binascii import unhexlify
from Cryptodome.Util.py3compat import bord, tobytes, is_bytes
from Cryptodome.Random import get_random_bytes
from . import cSHAKE128, SHA3_256
from .cSHAKE128 import _bytepad, _encode_str, _right_encode
class KMAC_Hash(object):
    """A KMAC hash object.
    Do not instantiate directly.
    Use the :func:`new` function.
    """

    def __init__(self, data, key, mac_len, custom, oid_variant, cshake, rate):
        self.oid = '2.16.840.1.101.3.4.2.' + oid_variant
        self.digest_size = mac_len
        self._mac = None
        partial_newX = _bytepad(_encode_str(tobytes(key)), rate)
        self._cshake = cshake._new(partial_newX, custom, b'KMAC')
        if data:
            self._cshake.update(data)

    def update(self, data):
        """Authenticate the next chunk of message.

        Args:
            data (bytes/bytearray/memoryview): The next chunk of the message to
            authenticate.
        """
        if self._mac:
            raise TypeError("You can only call 'digest' or 'hexdigest' on this object")
        self._cshake.update(data)
        return self

    def digest(self):
        """Return the **binary** (non-printable) MAC tag of the message.

        :return: The MAC tag. Binary form.
        :rtype: byte string
        """
        if not self._mac:
            self._cshake.update(_right_encode(self.digest_size * 8))
            self._mac = self._cshake.read(self.digest_size)
        return self._mac

    def hexdigest(self):
        """Return the **printable** MAC tag of the message.

        :return: The MAC tag. Hexadecimal encoded.
        :rtype: string
        """
        return ''.join(['%02x' % bord(x) for x in tuple(self.digest())])

    def verify(self, mac_tag):
        """Verify that a given **binary** MAC (computed by another party)
        is valid.

        Args:
          mac_tag (bytes/bytearray/memoryview): the expected MAC of the message.

        Raises:
            ValueError: if the MAC does not match. It means that the message
                has been tampered with or that the MAC key is incorrect.
        """
        secret = get_random_bytes(16)
        mac1 = SHA3_256.new(secret + mac_tag)
        mac2 = SHA3_256.new(secret + self.digest())
        if mac1.digest() != mac2.digest():
            raise ValueError('MAC check failed')

    def hexverify(self, hex_mac_tag):
        """Verify that a given **printable** MAC (computed by another party)
        is valid.

        Args:
            hex_mac_tag (string): the expected MAC of the message, as a hexadecimal string.

        Raises:
            ValueError: if the MAC does not match. It means that the message
                has been tampered with or that the MAC key is incorrect.
        """
        self.verify(unhexlify(tobytes(hex_mac_tag)))

    def new(self, **kwargs):
        """Return a new instance of a KMAC hash object.
        See :func:`new`.
        """
        if 'mac_len' not in kwargs:
            kwargs['mac_len'] = self.digest_size
        return new(**kwargs)