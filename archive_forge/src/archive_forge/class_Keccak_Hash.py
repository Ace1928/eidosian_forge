from Cryptodome.Util.py3compat import bord
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
class Keccak_Hash(object):
    """A Keccak hash object.
    Do not instantiate directly.
    Use the :func:`new` function.

    :ivar digest_size: the size in bytes of the resulting hash
    :vartype digest_size: integer
    """

    def __init__(self, data, digest_bytes, update_after_digest):
        self.digest_size = digest_bytes
        self._update_after_digest = update_after_digest
        self._digest_done = False
        self._padding = 1
        state = VoidPointer()
        result = _raw_keccak_lib.keccak_init(state.address_of(), c_size_t(self.digest_size * 2), c_ubyte(24))
        if result:
            raise ValueError('Error %d while instantiating keccak' % result)
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
            raise ValueError('Error %d while updating keccak' % result)
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
            raise ValueError('Error %d while squeezing keccak' % result)
        return get_raw_buffer(bfr)

    def hexdigest(self):
        """Return the **printable** digest of the message that has been hashed so far.

        :return: The hash digest, computed over the data processed so far.
                 Hexadecimal encoded.
        :rtype: string
        """
        return ''.join(['%02x' % bord(x) for x in self.digest()])

    def new(self, **kwargs):
        """Create a fresh Keccak hash object."""
        if 'digest_bytes' not in kwargs and 'digest_bits' not in kwargs:
            kwargs['digest_bytes'] = self.digest_size
        return new(**kwargs)