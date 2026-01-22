import abc
from collections import namedtuple
from ._error import (
from ._codec import decode_caveat
from ._macaroon import (
from ._versions import VERSION_2
from ._third_party import ThirdPartyCaveatInfo
import macaroonbakery.checkers as checkers
def _check_need_declared(ctx, cav_info, checker):
    arg = cav_info.condition
    i = arg.find(' ')
    if i <= 0:
        raise VerificationError('need-declared caveat requires an argument, got %q'.format(arg))
    need_declared = arg[0:i].split(',')
    for d in need_declared:
        if d == '':
            raise VerificationError('need-declared caveat with empty required attribute')
    if len(need_declared) == 0:
        raise VerificationError('need-declared caveat with no required attributes')
    cav_info = cav_info._replace(condition=arg[i + 1:])
    caveats = checker.check_third_party_caveat(ctx, cav_info)
    declared = {}
    for cav in caveats:
        if cav.location is not None and cav.location != '':
            continue
        try:
            cond, arg = checkers.parse_caveat(cav.condition)
        except ValueError:
            continue
        if cond != checkers.COND_DECLARED:
            continue
        parts = arg.split()
        if len(parts) != 2:
            raise VerificationError('declared caveat has no value')
        declared[parts[0]] = True
    for d in need_declared:
        if not declared.get(d, False):
            caveats.append(checkers.declared_caveat(d, ''))
    return caveats