import functools
@staticmethod
def _is_subnet_of(a, b):
    try:
        if a._version != b._version:
            raise TypeError(f'{a} and {b} are not of the same version')
        return b.network_address <= a.network_address and b.broadcast_address >= a.broadcast_address
    except AttributeError:
        raise TypeError(f'Unable to test subnet containment between {a} and {b}')