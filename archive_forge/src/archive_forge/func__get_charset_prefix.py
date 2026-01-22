import _sre
from . import _parser
from ._constants import *
from ._casefix import _EXTRA_CASES
def _get_charset_prefix(pattern, flags):
    while True:
        if not pattern.data:
            return None
        op, av = pattern.data[0]
        if op is not SUBPATTERN:
            break
        group, add_flags, del_flags, pattern = av
        flags = _combine_flags(flags, add_flags, del_flags)
        if flags & SRE_FLAG_IGNORECASE and flags & SRE_FLAG_LOCALE:
            return None
    iscased = _get_iscased(flags)
    if op is LITERAL:
        if iscased and iscased(av):
            return None
        return [(op, av)]
    elif op is BRANCH:
        charset = []
        charsetappend = charset.append
        for p in av[1]:
            if not p:
                return None
            op, av = p[0]
            if op is LITERAL and (not (iscased and iscased(av))):
                charsetappend((op, av))
            else:
                return None
        return charset
    elif op is IN:
        charset = av
        if iscased:
            for op, av in charset:
                if op is LITERAL:
                    if iscased(av):
                        return None
                elif op is RANGE:
                    if av[1] > 65535:
                        return None
                    if any(map(iscased, range(av[0], av[1] + 1))):
                        return None
        return charset
    return None