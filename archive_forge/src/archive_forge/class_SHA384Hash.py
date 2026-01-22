from Cryptodome.Util.py3compat import bord
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
class SHA384Hash(object):
    """A SHA-384 hash object.
    Do not instantiate directly. Use the :func:`new` function.

    :ivar oid: ASN.1 Object ID
    :vartype oid: string

    :ivar block_size: the size in bytes of the internal message block,
                      input to the compression function
    :vartype block_size: integer

    :ivar digest_size: the size in bytes of the resulting hash
    :vartype digest_size: integer
    """
    digest_size = 48
    block_size = 128
    oid = '2.16.840.1.101.3.4.2.2'

    def __init__(self, data=None):
        state = VoidPointer()
        result = _raw_sha384_lib.SHA384_init(state.address_of())
        if result:
            raise ValueError('Error %d while instantiating SHA384' % result)
        self._state = SmartPointer(state.get(), _raw_sha384_lib.SHA384_destroy)
        if data:
            self.update(data)

    def update(self, data):
        """Continue hashing of a message by consuming the next chunk of data.

        Args:
            data (byte string/byte array/memoryview): The next chunk of the message being hashed.
        """
        result = _raw_sha384_lib.SHA384_update(self._state.get(), c_uint8_ptr(data), c_size_t(len(data)))
        if result:
            raise ValueError('Error %d while hashing data with SHA384' % result)

    def digest(self):
        """Return the **binary** (non-printable) digest of the message that has been hashed so far.

        :return: The hash digest, computed over the data processed so far.
                 Binary form.
        :rtype: byte string
        """
        bfr = create_string_buffer(self.digest_size)
        result = _raw_sha384_lib.SHA384_digest(self._state.get(), bfr, c_size_t(self.digest_size))
        if result:
            raise ValueError('Error %d while making SHA384 digest' % result)
        return get_raw_buffer(bfr)

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
        clone = SHA384Hash()
        result = _raw_sha384_lib.SHA384_copy(self._state.get(), clone._state.get())
        if result:
            raise ValueError('Error %d while copying SHA384' % result)
        return clone

    def new(self, data=None):
        """Create a fresh SHA-384 hash object."""
        return SHA384Hash(data)