from Cryptodome.Util._raw_api import (VoidPointer, SmartPointer,
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Util.py3compat import bchr
from .keccak import _raw_keccak_lib
class TurboSHAKE(object):
    """A TurboSHAKE hash object.
    Do not instantiate directly.
    Use the :func:`new` function.
    """

    def __init__(self, capacity, domain_separation, data):
        state = VoidPointer()
        result = _raw_keccak_lib.keccak_init(state.address_of(), c_size_t(capacity), c_ubyte(12))
        if result:
            raise ValueError('Error %d while instantiating TurboSHAKE' % result)
        self._state = SmartPointer(state.get(), _raw_keccak_lib.keccak_destroy)
        self._is_squeezing = False
        self._capacity = capacity
        self._domain = domain_separation
        if data:
            self.update(data)

    def update(self, data):
        """Continue hashing of a message by consuming the next chunk of data.

        Args:
            data (byte string/byte array/memoryview): The next chunk of the message being hashed.
        """
        if self._is_squeezing:
            raise TypeError("You cannot call 'update' after the first 'read'")
        result = _raw_keccak_lib.keccak_absorb(self._state.get(), c_uint8_ptr(data), c_size_t(len(data)))
        if result:
            raise ValueError('Error %d while updating TurboSHAKE state' % result)
        return self

    def read(self, length):
        """
        Compute the next piece of XOF output.

        .. note::
            You cannot use :meth:`update` anymore after the first call to
            :meth:`read`.

        Args:
            length (integer): the amount of bytes this method must return

        :return: the next piece of XOF output (of the given length)
        :rtype: byte string
        """
        self._is_squeezing = True
        bfr = create_string_buffer(length)
        result = _raw_keccak_lib.keccak_squeeze(self._state.get(), bfr, c_size_t(length), c_ubyte(self._domain))
        if result:
            raise ValueError('Error %d while extracting from TurboSHAKE' % result)
        return get_raw_buffer(bfr)

    def new(self, data=None):
        return type(self)(self._capacity, self._domain, data)

    def _reset(self):
        result = _raw_keccak_lib.keccak_reset(self._state.get())
        if result:
            raise ValueError('Error %d while resetting TurboSHAKE state' % result)
        self._is_squeezing = False