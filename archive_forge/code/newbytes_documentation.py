from numbers import Integral
import string
import copy
from future.utils import istext, isbytes, PY2, PY3, with_metaclass
from future.types import no, issubset
from future.types.newobject import newobject

        B.maketrans(frm, to) -> translation table

        Return a translation table (a bytes object of length 256) suitable
        for use in the bytes or bytearray translate method where each byte
        in frm is mapped to the byte at the same position in to.
        The bytes objects frm and to must be of the same length.
        