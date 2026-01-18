from Cryptodome import Random
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Util.py3compat import iter_range
from Cryptodome.Util.number import sieve_base as _sieve_base_large
def test_probable_prime(candidate, randfunc=None):
    """Test if a number is prime.

    A number is qualified as prime if it passes a certain
    number of Miller-Rabin tests (dependent on the size
    of the number, but such that probability of a false
    positive is less than 10^-30) and a single Lucas test.

    For instance, a 1024-bit candidate will need to pass
    4 Miller-Rabin tests.

    :Parameters:
      candidate : integer
        The number to test for primality.
      randfunc : callable
        The routine to draw random bytes from to select Miller-Rabin bases.
    :Returns:
      ``PROBABLE_PRIME`` if the number if prime with very high probability.
      ``COMPOSITE`` if the number is a composite.
      For efficiency reasons, ``COMPOSITE`` is also returned for small primes.
    """
    if randfunc is None:
        randfunc = Random.new().read
    if not isinstance(candidate, Integer):
        candidate = Integer(candidate)
    if int(candidate) in _sieve_base:
        return PROBABLY_PRIME
    try:
        map(candidate.fail_if_divisible_by, _sieve_base)
    except ValueError:
        return COMPOSITE
    mr_ranges = ((220, 30), (280, 20), (390, 15), (512, 10), (620, 7), (740, 6), (890, 5), (1200, 4), (1700, 3), (3700, 2))
    bit_size = candidate.size_in_bits()
    try:
        mr_iterations = list(filter(lambda x: bit_size < x[0], mr_ranges))[0][1]
    except IndexError:
        mr_iterations = 1
    if miller_rabin_test(candidate, mr_iterations, randfunc=randfunc) == COMPOSITE:
        return COMPOSITE
    if lucas_test(candidate) == COMPOSITE:
        return COMPOSITE
    return PROBABLY_PRIME