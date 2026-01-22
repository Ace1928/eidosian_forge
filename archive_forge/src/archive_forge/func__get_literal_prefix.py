import _sre
from . import _parser
from ._constants import *
from ._casefix import _EXTRA_CASES
def _get_literal_prefix(pattern, flags):
    prefix = []
    prefixappend = prefix.append
    prefix_skip = None
    iscased = _get_iscased(flags)
    for op, av in pattern.data:
        if op is LITERAL:
            if iscased and iscased(av):
                break
            prefixappend(av)
        elif op is SUBPATTERN:
            group, add_flags, del_flags, p = av
            flags1 = _combine_flags(flags, add_flags, del_flags)
            if flags1 & SRE_FLAG_IGNORECASE and flags1 & SRE_FLAG_LOCALE:
                break
            prefix1, prefix_skip1, got_all = _get_literal_prefix(p, flags1)
            if prefix_skip is None:
                if group is not None:
                    prefix_skip = len(prefix)
                elif prefix_skip1 is not None:
                    prefix_skip = len(prefix) + prefix_skip1
            prefix.extend(prefix1)
            if not got_all:
                break
        else:
            break
    else:
        return (prefix, prefix_skip, True)
    return (prefix, prefix_skip, False)