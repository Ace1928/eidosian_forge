import abc
from Cryptodome.Util.py3compat import iter_range, bord, bchr, ABC
from Cryptodome import Random
Multiply two integers, take the modulo, and encode as big endian.
        This specialized method is used for RSA decryption.

        Args:
          term1 : integer
            The first term of the multiplication, non-negative.
          term2 : integer
            The second term of the multiplication, non-negative.
          modulus: integer
            The modulus, a positive odd number.
        :Returns:
            A byte string, with the result of the modular multiplication
            encoded in big endian mode.
            It is as long as the modulus would be, with zero padding
            on the left if needed.
        