from numbers import Number
from future.utils import PY3, istext, with_metaclass, isnewbytes
from future.types import no, issubset
from future.types.newobject import newobject

        S.translate(table) -> str

        Return a copy of the string S, where all characters have been mapped
        through the given translation table, which must be a mapping of
        Unicode ordinals to Unicode ordinals, strings, or None.
        Unmapped characters are left untouched. Characters mapped to None
        are deleted.
        