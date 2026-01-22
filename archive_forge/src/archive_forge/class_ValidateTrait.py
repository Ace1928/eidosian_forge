from enum import IntEnum
import traits.ctraits
class ValidateTrait(IntEnum):
    """ These are indices into the ctraits.c validate_handlers array. """
    type = 0
    instance = 1
    self_type = 2
    int_range = 3
    float_range = 4
    enum = 5
    map = 6
    complex = 7
    slow = 8
    tuple = 9
    prefix_map = 10
    coerce = 11
    cast = 12
    function = 13
    python = 14
    adapt = 19
    int = 20
    float = 21
    callable = 22