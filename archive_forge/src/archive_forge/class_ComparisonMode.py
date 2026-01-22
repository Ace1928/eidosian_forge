from enum import IntEnum
import traits.ctraits
class ComparisonMode(IntEnum):
    """ Comparison mode.

    Indicates when trait change notifications should be generated based upon
    the result of comparing the old and new values of a trait assignment:

    Enumeration members:

    none
        The values are not compared and a trait change notification is
        generated on each assignment.
    identity
        A trait change notification is generated if the old and new values are
        not the same object.
    equality
        A trait change notification is generated if the old and new values are
        not the same object, and not equal using Python's standard equality
        testing. This is the default.
    """
    none = 0
    identity = 1
    equality = 2