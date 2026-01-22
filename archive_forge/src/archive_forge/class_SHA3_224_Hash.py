from Cryptodome.Util.py3compat import bord
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
from Cryptodome.Hash.keccak import _raw_keccak_lib
class SHA3_224_Hash(object):
    """A SHA3-224 hash object.
    Do not instantiate directly.
    Use the :func:`new` function.

    :ivar oid: ASN.1 Object ID
    :vartype oid: string

    :ivar digest_size: the size in bytes of the resulting hash
    :vartype digest_size: integer
    """
    digest_size = 28
    oid = '2.16.840.1.101.3.4.2.7'
    block_size = 144

    def __init__(self, data, update_after_digest):
        self._update_after_digest = update_after_digest
        self._digest_done = False
        self._padding = 6
        state = VoidPointer()
        result = _raw_keccak_lib.keccak_init(state.address_of(), c_size_t(self.digest_size * 2), c_ubyte(24))
        if result:
            raise ValueError('Error %d while instantiating SHA-3/224' % result)
        self._state = SmartPointer(state.get(), _raw_keccak_lib.keccak_destroy)
        if data:
            self.update(data)

    def update(self, data):
        """Continue hashing of a message by consuming the next chunk of data.

        Args:
            data (byte string/byte array/memoryview): The next chunk of the message being hashed.
        """
        if self._digest_done and (not self._update_after_digest):
            raise TypeError("You can only call 'digest' or 'hexdigest' on this object")
        result = _raw_keccak_lib.keccak_absorb(self._state.get(), c_uint8_ptr(data), c_size_t(len(data)))
        if result:
            raise ValueError('Error %d while updating SHA-3/224' % result)
        return self

    def digest(self):
        """Return the **binary** (non-printable) digest of the message that has been hashed so far.

        :return: The hash digest, computed over the data processed so far.
                 Binary form.
        :rtype: byte string
        """
        self._digest_done = True
        bfr = create_string_buffer(self.digest_size)
        result = _raw_keccak_lib.keccak_digest(self._state.get(), bfr, c_size_t(self.digest_size), c_ubyte(self._padding))
        if result:
            raise ValueError('Error %d while instantiating SHA-3/224' % result)
        self._digest_value = get_raw_buffer(bfr)
        return self._digest_value

    def hexdigest(self):
        """Return the **printable** digest of the message that has been hashed so far.

        :return: The hash digest, computed over the data processed so far.
                 Hexadecimal encoded.
        :rtype: string
        """
        return ''.join(['%02x' % bord(x) for x in self.digest()])

    def copy(self):
        """Return a copy ("clone") of the hash object.

        The copy will have the same internal state as the original hash
        object.
        This can be used to efficiently compute the digests of strings that
        share a common initial substring.

        :return: A hash object of the same type
        """
        clone = self.new()
        result = _raw_keccak_lib.keccak_copy(self._state.get(), clone._state.get())
        if result:
            raise ValueError('Error %d while copying SHA3-224' % result)
        return clone

    def new(self, data=None):
        """Create a fresh SHA3-224 hash object."""
        return type(self)(data, self._update_after_digest)