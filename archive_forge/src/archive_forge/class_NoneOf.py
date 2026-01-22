from .core import Adapter, AdaptationError, Pass
from .lib import int_to_bin, bin_to_int, swap_bytes
from .lib import FlagsContainer, HexString
from .lib.py3compat import BytesIO, decodebytes
class NoneOf(Validator):
    """
    Validates that the object is none of the listed values.

    :param ``Construct`` subcon: object to validate
    :param iterable invalids: a set of invalid values

    >>> NoneOf(UBInt8("foo"), [4,5,6,7]).parse("\\x08")
    8
    >>> NoneOf(UBInt8("foo"), [4,5,6,7]).parse("\\x06")
    Traceback (most recent call last):
        ...
    construct.core.ValidationError: ('invalid object', 6)
    """
    __slots__ = ['invalids']

    def __init__(self, subcon, invalids):
        Validator.__init__(self, subcon)
        self.invalids = invalids

    def _validate(self, obj, context):
        return obj not in self.invalids