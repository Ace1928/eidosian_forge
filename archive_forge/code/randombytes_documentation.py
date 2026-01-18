from nacl import exceptions as exc
from nacl._sodium import ffi, lib

    Returns ``size`` number of deterministically generated pseudorandom bytes
    from a seed

    :param size: int
    :param seed: bytes
    :rtype: bytes
    