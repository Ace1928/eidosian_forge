import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
class verify:
    """
    Check an enumeration for various constraints. (see EnumCheck)
    """

    def __init__(self, *checks):
        self.checks = checks

    def __call__(self, enumeration):
        checks = self.checks
        cls_name = enumeration.__name__
        if Flag is not None and issubclass(enumeration, Flag):
            enum_type = 'flag'
        elif issubclass(enumeration, Enum):
            enum_type = 'enum'
        else:
            raise TypeError("the 'verify' decorator only works with Enum and Flag")
        for check in checks:
            if check is UNIQUE:
                duplicates = []
                for name, member in enumeration.__members__.items():
                    if name != member.name:
                        duplicates.append((name, member.name))
                if duplicates:
                    alias_details = ', '.join(['%s -> %s' % (alias, name) for alias, name in duplicates])
                    raise ValueError('aliases found in %r: %s' % (enumeration, alias_details))
            elif check is CONTINUOUS:
                values = set((e.value for e in enumeration))
                if len(values) < 2:
                    continue
                low, high = (min(values), max(values))
                missing = []
                if enum_type == 'flag':
                    for i in range(_high_bit(low) + 1, _high_bit(high)):
                        if 2 ** i not in values:
                            missing.append(2 ** i)
                elif enum_type == 'enum':
                    for i in range(low + 1, high):
                        if i not in values:
                            missing.append(i)
                else:
                    raise Exception('verify: unknown type %r' % enum_type)
                if missing:
                    raise ValueError(('invalid %s %r: missing values %s' % (enum_type, cls_name, ', '.join((str(m) for m in missing))))[:256])
            elif check is NAMED_FLAGS:
                member_names = enumeration._member_names_
                member_values = [m.value for m in enumeration]
                missing_names = []
                missing_value = 0
                for name, alias in enumeration._member_map_.items():
                    if name in member_names:
                        continue
                    if alias.value < 0:
                        continue
                    values = list(_iter_bits_lsb(alias.value))
                    missed = [v for v in values if v not in member_values]
                    if missed:
                        missing_names.append(name)
                        missing_value |= reduce(_or_, missed)
                if missing_names:
                    if len(missing_names) == 1:
                        alias = 'alias %s is missing' % missing_names[0]
                    else:
                        alias = 'aliases %s and %s are missing' % (', '.join(missing_names[:-1]), missing_names[-1])
                    if _is_single_bit(missing_value):
                        value = 'value 0x%x' % missing_value
                    else:
                        value = 'combined values of 0x%x' % missing_value
                    raise ValueError('invalid Flag %r: %s %s [use enum.show_flag_values(value) for details]' % (cls_name, alias, value))
        return enumeration