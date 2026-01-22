import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
@classmethod
def _check_for_existing_members_(mcls, class_name, bases):
    for chain in bases:
        for base in chain.__mro__:
            if isinstance(base, EnumType) and base._member_names_:
                raise TypeError('<enum %r> cannot extend %r' % (class_name, base))