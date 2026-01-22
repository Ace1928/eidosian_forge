import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
def _create_(cls, class_name, names, *, module=None, qualname=None, type=None, start=1, boundary=None):
    """
        Convenience method to create a new Enum class.

        `names` can be:

        * A string containing member names, separated either with spaces or
          commas.  Values are incremented by 1 from `start`.
        * An iterable of member names.  Values are incremented by 1 from `start`.
        * An iterable of (member name, value) pairs.
        * A mapping of member name -> value pairs.
        """
    metacls = cls.__class__
    bases = (cls,) if type is None else (type, cls)
    _, first_enum = cls._get_mixins_(class_name, bases)
    classdict = metacls.__prepare__(class_name, bases)
    if isinstance(names, str):
        names = names.replace(',', ' ').split()
    if isinstance(names, (tuple, list)) and names and isinstance(names[0], str):
        original_names, names = (names, [])
        last_values = []
        for count, name in enumerate(original_names):
            value = first_enum._generate_next_value_(name, start, count, last_values[:])
            last_values.append(value)
            names.append((name, value))
    if names is None:
        names = ()
    for item in names:
        if isinstance(item, str):
            member_name, member_value = (item, names[item])
        else:
            member_name, member_value = item
        classdict[member_name] = member_value
    if module is None:
        try:
            module = sys._getframe(2).f_globals['__name__']
        except (AttributeError, ValueError, KeyError):
            pass
    if module is None:
        _make_class_unpicklable(classdict)
    else:
        classdict['__module__'] = module
    if qualname is not None:
        classdict['__qualname__'] = qualname
    return metacls.__new__(metacls, class_name, bases, classdict, boundary=boundary)