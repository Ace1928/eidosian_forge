from Cryptodome.Util.number import long_to_bytes
from Cryptodome.Util.py3compat import bchr
from . import TurboSHAKE128
class K12_XOF(object):
    """A KangarooTwelve hash object.
    Do not instantiate directly.
    Use the :func:`new` function.
    """

    def __init__(self, data, custom):
        if custom == None:
            custom = b''
        self._custom = custom + _length_encode(len(custom))
        self._state = SHORT_MSG
        self._padding = None
        self._hash1 = TurboSHAKE128.new(domain=1)
        self._length1 = 0
        self._hash2 = None
        self._length2 = 0
        self._ctr = 0
        if data:
            self.update(data)

    def update(self, data):
        """Hash the next piece of data.

        .. note::
            For better performance, submit chunks with a length multiple of 8192 bytes.

        Args:
            data (byte string/byte array/memoryview): The next chunk of the
              message to hash.
        """
        if self._state == SQUEEZING:
            raise TypeError("You cannot call 'update' after the first 'read'")
        if self._state == SHORT_MSG:
            next_length = self._length1 + len(data)
            if next_length + len(self._custom) <= 8192:
                self._length1 = next_length
                self._hash1.update(data)
                return self
            self._state = LONG_MSG_S0
        if self._state == LONG_MSG_S0:
            data_mem = memoryview(data)
            assert self._length1 < 8192
            dtc = min(len(data), 8192 - self._length1)
            self._hash1.update(data_mem[:dtc])
            self._length1 += dtc
            if self._length1 < 8192:
                return self
            assert self._length1 == 8192
            divider = b'\x03' + b'\x00' * 7
            self._hash1.update(divider)
            self._length1 += 8
            self._hash2 = TurboSHAKE128.new(domain=11)
            self._length2 = 0
            self._ctr = 1
            self._state = LONG_MSG_SX
            return self.update(data_mem[dtc:])
        assert self._state == LONG_MSG_SX
        index = 0
        len_data = len(data)
        data_mem = memoryview(data)
        while index < len_data:
            new_index = min(index + 8192 - self._length2, len_data)
            self._hash2.update(data_mem[index:new_index])
            self._length2 += new_index - index
            index = new_index
            if self._length2 == 8192:
                cv_i = self._hash2.read(32)
                self._hash1.update(cv_i)
                self._length1 += 32
                self._hash2._reset()
                self._length2 = 0
                self._ctr += 1
        return self

    def read(self, length):
        """
        Produce more bytes of the digest.

        .. note::
            You cannot use :meth:`update` anymore after the first call to
            :meth:`read`.

        Args:
            length (integer): the amount of bytes this method must return

        :return: the next piece of XOF output (of the given length)
        :rtype: byte string
        """
        custom_was_consumed = False
        if self._state == SHORT_MSG:
            self._hash1.update(self._custom)
            self._padding = 7
            self._state = SQUEEZING
        if self._state == LONG_MSG_S0:
            self.update(self._custom)
            custom_was_consumed = True
            assert self._state == LONG_MSG_SX
        if self._state == LONG_MSG_SX:
            if not custom_was_consumed:
                self.update(self._custom)
            if self._length2 > 0:
                cv_i = self._hash2.read(32)
                self._hash1.update(cv_i)
                self._length1 += 32
                self._hash2._reset()
                self._length2 = 0
                self._ctr += 1
            trailer = _length_encode(self._ctr - 1) + b'\xff\xff'
            self._hash1.update(trailer)
            self._padding = 6
            self._state = SQUEEZING
        self._hash1._domain = self._padding
        return self._hash1.read(length)

    def new(self, data=None, custom=b''):
        return type(self)(data, custom)