from rsa._compat import range
import rsa.common
import rsa.randnum
def are_relatively_prime(a, b):
    """Returns True if a and b are relatively prime, and False if they
    are not.

    >>> are_relatively_prime(2, 3)
    True
    >>> are_relatively_prime(2, 4)
    False
    """
    d = gcd(a, b)
    return d == 1