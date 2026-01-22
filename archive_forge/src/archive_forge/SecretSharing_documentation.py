from Cryptodome.Util.py3compat import is_native_int
from Cryptodome.Util import number
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Random import get_random_bytes as rng
Recombine a secret, if enough shares are presented.

        Args:
          shares (tuples):
            The *k* tuples, each containin the index (an integer) and
            the share (a byte string, 16 bytes long) that were assigned to
            a participant.
          ssss (bool):
            If ``True``, the shares were produced by the ``ssss`` utility.
            Default: ``False``.

        Return:
            The original secret, as a byte string (16 bytes long).
        