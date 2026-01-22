from enum import IntEnum
import traits.ctraits
class TraitKind(IntEnum):
    """ These determine the getters and setters used by the cTrait instance.
    """
    trait = 0
    python = 1
    event = 2
    delegate = 3
    property = 4
    disallow = 5
    read_only = 6
    constant = 7
    generic = 8